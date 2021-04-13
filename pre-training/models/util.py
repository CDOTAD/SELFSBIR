import torch
import torch.distributed as dist


def dist_collect(x):
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


class DistributedShufle:
    @staticmethod
    def foward_shuffle(x, epoch):
        x_all = dist_collect(x)
        forward_inds, backward_inds = DistributedShufle.get_shuffle_ids(x_all.shape[0], epoch)
        forward_inds_label = DistributedShufle.get_local_id(forward_inds)

        return x_all[forward_inds_label], backward_inds

    @staticmethod
    def backward_shuffle(x, backward_inds, return_local=True):
        x_all = dist_collect(x)
        if return_local:
            backward_inds_local = DistributedShufle.get_local_id(backward_inds)
            return x_all[backward_inds], x_all[backward_inds_local]
        else:
            return x_all[backward_inds]
    @staticmethod
    def get_local_id(ids):
        return ids.chunk(dist.get_world_size())[dist.get_rank()]

    @staticmethod
    def get_shuffle_ids(bsz, epoch):
        torch.manual_seed(epoch)

        forward_inds = torch.randperm(bsz).long().cuda()

        backward_inds = torch.zeros(forward_inds.shape[0]).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)

        return forward_inds, backward_inds


def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)


def moment_updata(model, model_ema, m):
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)