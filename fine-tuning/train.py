import torch
from data import DatasetCreator
from torch import nn
import os
from torchnet.meter import AverageValueMeter
from tensorboardX import SummaryWriter
from utils.tester import Tester
from models.TripletLoss import TripletLoss
import numpy as np
import random
import time
import math
import yaml
import torch.utils.data
import shutil


try:
    from apex import amp
except ImportError:
    amp = None


class Config(object):
    def __init__(self):
        return


def adjust_learning_rate_cosine(step):
    if step < warmup_epochs:
        cur_lr = step / 10 * init_lr
    else:
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (step - warmup_epochs) / (epochs - warmup_epochs + 1)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


if __name__ == '__main__':
    CONFIG_DIR = '.'
    CONFIG_NAME = 'rnd_mix_edge_chair'
    CONFIG_PATH = os.path.join(CONFIG_DIR, CONFIG_NAME+'.yaml')

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = f.read()

    config = yaml.load(config, Loader=yaml.FullLoader)

    devices = config['train']['devices']
    devices_str = ''
    for device in devices:
        devices_str += str(device) + ', '
    devices_str = devices_str[:-2]
    os.environ['CUDA_VISIBLE_DEVICES'] = devices_str

    use_amp = config['amp']['use']
    opt_level = config['amp']['opt_level']
    if use_amp:
        try:
            from apex import amp
        except ImportError:
            amp = None
            use_amp = False
    set_seed = config['seed']['use']
    seed = config['seed']['seed']
    if set_seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # save_path = {root}/{dbname}/{exp_name}
    # tf_event_root = {save_path}/*.event
    # saved_ckpt = {save_path}/*.pth
    # log = {save_path}/*.log
    # config_file = {save_path}/*.yaml

    LOG_DIR = config['log']['log_dir']
    dataname = config['train']['dataset']
    exp_name = config['log']['exp_name']

    SAVE_DIR = os.path.join(LOG_DIR, dataname, exp_name)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    save_model = config['model']['save_model']

    writer = SummaryWriter(SAVE_DIR)
    shutil.copy(CONFIG_PATH, os.path.join(SAVE_DIR, CONFIG_NAME+'.yaml'))

    pre_train_type = config['model']['pre_train_type']
    CKPT_DIR = config['model']['ckpt_dir']
    state_dic = torch.load(CKPT_DIR, map_location=torch.device('cpu'))

    original_resnet = False
    cls_pre_train = False
    if pre_train_type != 'my':
        original_resnet = True
    if pre_train_type == 'cls':
        cls_pre_train = True

    if original_resnet:
        from models.sketch_resnet import resnet50

        model = resnet50(pretrained=True)
        # model.fc = nn.Linear(2048, 125)
        del model.fc
        model.fc = lambda x: x
        if cls_pre_train:
            model = model.cuda()
        else:
            model = nn.DataParallel(model)
            if 'module' in list(state_dic.keys())[0]:
                model.load_state_dict(state_dic, strict=False)
            else:
                model.module.load_state_dict(state_dic, strict=False)
            model = model.module
            model.cuda()
    else:
        from models.resnet import resnet50
        model = resnet50(output_layer=6, stage='finetune')
        model = nn.DataParallel(model)
        model.load_state_dict(state_dic['model'], strict=False)
        del model.module.fc
        model.module.fc = lambda x: x
        model = model.module
        model = model.cuda()

    test_f = config['test']['test_f']
    sketch_test = config['test']['sketch_root']
    photo_test = config['test']['photo_root']

    batch_size = config['train']['batch_size']
    data_opt = Config()
    data_opt.photo_root = config['train']['photo_root']
    data_opt.sketch_root = config['train']['sketch_root']

    dataset = DatasetCreator.getDataset(dataname)
    train_dataset = dataset(data_opt.photo_root, data_opt.sketch_root)
    sketchydataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    my_triplet_loss = TripletLoss().cuda()

    init_lr = config['train']['learning_rate']
    adj_lr = config['train']['adj_lr']
    epochs = config['train']['epochs']
    warmup_epochs = config['train']['warmup_epochs']
    using_sgd = config['train']['using_sgd']
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9,
                                weight_decay=1e-4) if using_sgd else torch.optim.Adam(model.parameters(),
                                                                                      lr=init_lr)  # , weight_decay=1e-4)

    if use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
        my_triplet_loss = amp.initialize(my_triplet_loss, opt_level=opt_level)

    '''
    if use_amp:
        [p_net, s_net], [p_optimizer, s_optimizer] = amp.initialize([p_net, s_net], [p_optimizer, s_optimizer], opt_level=opt_level)
        [photo_cat_loss, sketch_cat_loss, my_triplet_loss] = amp.initialize([photo_cat_loss, sketch_cat_loss, my_triplet_loss], opt_level=opt_level)
    schedule = [int(0.6*epochs), int(0.8*epochs)]
    '''
    triplet_loss_meter = AverageValueMeter()

    best_recall1 = 0
    best_recall5 = 0
    best_recall10 = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        adjust_learning_rate_cosine(epoch)
        cur_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/lr', cur_lr, epoch)
        for ii, data in enumerate(sketchydataset):
            optimizer.zero_grad()

            photo = data['P'].cuda()
            sketch = data['S'].cuda()
            label = data['L'].cuda()
            bs = photo.size(0)

            photo_sketch = torch.cat([photo, sketch], dim=0)

            cat, feature = model(photo_sketch)

            p_cat, s_cat = torch.split(cat, [bs, bs], dim=0)
            p_feature, s_feature = torch.split(feature, [bs, bs], dim=0)

            tri_loss = my_triplet_loss(s_feature, p_feature)

            triplet_loss_meter.add(tri_loss.item())
            if use_amp:
                with amp.scale_loss(tri_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                tri_loss.backward()
            optimizer.step()
        end_time = time.time()
        print('======================== epoch : {0} ============================'.format(epoch))
        print('triplet_loss: {0}        time cost: {1}'.format(triplet_loss_meter.value()[0], end_time - start_time))
        writer.add_scalar('Train/Triplet', triplet_loss_meter.value()[0], epoch)

        triplet_loss_meter.reset()

        if epoch % test_f == 0:
            model.eval()
            
            test_config = Config()
            test_config.batch_size = config['test']['batch_size']
            test_config.photo_net = model.eval()
            test_config.sketch_net = model.eval()

            test_config.photo_test = photo_test
            test_config.sketch_test = sketch_test
            
            test_start_time = time.time()
            tester = Tester(test_config)
            test_result = tester.test(dbname=dataname)
            test_end_time = time.time()
            print('test time :', test_end_time - test_start_time)
            result_key = list(test_result.keys())
            if test_result[result_key[0]] > best_recall1:
                best_recall1 = test_result[result_key[0]]
                best_recall5 = test_result[result_key[1]]
                best_recall10 = test_result[result_key[2]]
                best_epoch = epoch
                
                if save_model:
                    torch.save(model.state_dict(), SAVE_DIR + '/best.pth')

            writer.add_scalar('Test/{0}'.format(result_key[0]), test_result[result_key[0]], epoch)
            writer.add_scalar('Test/{0}'.format(result_key[1]), test_result[result_key[1]], epoch)
            writer.add_scalar('Test/{0}'.format(result_key[2]), test_result[result_key[2]], epoch)

            print('best epoch: ', best_epoch, '         best recall@1 :', best_recall1)
            print('recall@5 :', best_recall5, '         best recall@10 :', best_recall10)
            # exit(0)




