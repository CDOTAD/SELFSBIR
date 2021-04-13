from data import ImageEdgeMoCo
import torch

from torch.utils.tensorboard import SummaryWriter

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from models.Contrast import MemoryMoCo
from models.NCECriterion import NCESoftmaxLoss
from models.resnet import resnet50
from models.util import moment_updata
from models.util import set_bn_train
from models.util import DistributedShufle

from lr_scheduler import get_scheduler

from torchnet.meter import AverageValueMeter
import os
import numpy as np
import json
import logging
import visdom

from logger import setup_logger

import argparse


def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_root', type=str)
    parser.add_argument('--edge_root', type=str)

    parser.add_argument('--dataset', type=str)

    parser.add_argument('--crop', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)

    parser.add_argument('--alpha', type=float)
    parser.add_argument('--nce_k', type=int)
    parser.add_argument('--nce_t', type=float)

    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--base-learning-rate', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    # warm up
    parser.add_argument('--warmup_epoch', type=int)
    parser.add_argument('--warmup_multiplier', type=int, default=100)
    # multi step decay
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200],  nargs='+')

    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=10)

    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--rng_seed', type=int, default=0)
    
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='./output')

    args = parser.parse_args()

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)

    return args


def get_loader(args):
    # train_dataset = SketchyMoCo(image_root=args.img_root, edge_root=args.edge_root)
    train_dataset = ImageEdgeMoCo(image_root=args.img_root, edge_root=args.edge_root)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)

    return train_loader


def build_model():
    model = resnet50().cuda()
    model_ema = resnet50().cuda()

    moment_updata(model, model_ema, 0)
    return model, model_ema


def load_checkpoint(args, model, model_ema, contrast, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.resume))

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])
    contrast.load_state_dict(checkpoint['contrast'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, model_ema, contrast, optimizer, scheduler):
    state = {
        'opt': args,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    torch.save(state, os.path.join(args.output_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth'))


def main(args):

    # vis = visdom.Visdom(env='sketchymoco')
    vis = None

    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)

    model, model_ema = build_model()
    contrast = MemoryMoCo(128, args.nce_k, args.nce_t).cuda()
    # print(args.batch_size*dist.get_world_size()/256 * args.base_learning_rate)
    # print(dist.get_world_size())
    criterion = NCESoftmaxLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.batch_size*dist.get_world_size()/256 * args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, model_ema, contrast, optimizer, scheduler)

    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        loss, prob, acc1, acc5 = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, scheduler, args, vis)

        if summary_writer is not None:
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('ins_prob', prob, epoch)
            summary_writer.add_scalar('learning_rater', optimizer.param_groups[0]['lr'], epoch)
            summary_writer.add_scalar('acc1', acc1, epoch)
            summary_writer.add_scalar('acc5', acc5, epoch)

        if dist.get_rank() == 0:
            save_checkpoint(args, epoch, model, model_ema, contrast, optimizer, scheduler)


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, scheduler, args, vis=None):

    model.train()
    set_bn_train(model_ema)

    loss_meter = AverageValueMeter()
    prob_meter = AverageValueMeter()
    top1 = AverageValueMeter()
    top5 = AverageValueMeter()

    for idx, inputs in enumerate(train_loader):
        bsz = inputs.size(0)
        x1, x2 = torch.split(inputs, [3, 3], dim=1)
        x1.contiguous()
        x2.contiguous()
        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)
        if vis and dist.get_rank() == 0:
            vis.images(x1.detach().cpu().numpy(), win='x1')
            vis.images(x2.detach().cpu().numpy(), win='x2')
        feat_q = model(x1)
        with torch.no_grad():
            x2_shuffled, bacward_inds = DistributedShufle.foward_shuffle(x2, epoch)
            feat_k = model_ema(x2_shuffled)
            feat_k_all, feat_k = DistributedShufle.backward_shuffle(feat_k, bacward_inds, return_local=True)

        out = contrast(feat_q, feat_k, feat_k_all)
        target = torch.zeros([out.shape[0]]).long().to(out.device)
        loss = criterion(out, target)
        prob = F.softmax(out, dim=1)[:, 0].mean()

        acc1, acc5 = accuracy(out, target, topk=(1, 5))
        top1.add(acc1.item())
        top5.add(acc5.item())

        optimizer.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        moment_updata(model, model_ema, args.alpha)

        loss_meter.add(loss.item())
        prob_meter.add(prob.item())

        if idx % args.print_freq == 0:
            logger.info(f'Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                        f'loss {loss_meter.val:.3f} ({loss_meter.value()[0]:.3f})\t'
                        f'prob {prob_meter.val:.3f} ({prob_meter.value()[0]:.3f})\t'
                        f'acc1 {top1.val:.3f} ({top1.value()[0]:.3f})\t'
                        f'acc5 {top5.val:.3f} ({top5.value()[0]:.3f})')

    return loss_meter.value()[0], prob_meter.value()[0], top1.value()[0], top5.value()[0]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    opt = parse_option()
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name='moco')
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)

    main(opt)
