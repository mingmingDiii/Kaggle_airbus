#!/usr/bin/env python

"""
Stochastic Weight Averaging (SWA)

Averaging Weights Leads to Wider Optima and Better Generalization

https://github.com/timgaripov/swa
"""
import torch
from main_code_seg.model_include import *
from tqdm import tqdm
import numpy as np
from main_code_seg.data_lib import torch_data_load
from torch import nn
from torch.autograd import Variable
SEED = 1234

def moving_average(net1, net2, alpha=1.):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    pbar = tqdm(loader, unit="images", unit_scale=loader.batch_size)
    for batch in pbar:
        input, targets = batch[0], batch[1]
        input = Variable(input).cuda()
        b = input.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        epoch = os.path.basename(snapshot).split('_')[-1]
        epoch = int(epoch)+1
        net.load_state_dict(torch.load(snapshot))


    net = net.cuda()
    return net, epoch


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from torchvision.transforms import Compose
    from torch.utils.data import DataLoader
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    swa_exp = 'smixa0a1'
    swa_path = 'experiments/swa/{}'.format(swa_exp)
    if not os.path.exists(swa_path):
        os.mkdir(swa_path)

    net, _ = build_network(None, 'dense_normal')
    net2, _ = build_network(None, 'dense_normal')

    files = [
        '/mnt/sda1/don/documents/airbus/main_code_seg/experiments/dn_SLovDiB768_has_norm/a0/weights/busnet_29',
        '/mnt/sda1/don/documents/airbus/main_code_seg/experiments/dn_SLovDiB768_has_norm/a1/weights/busnet_19'
    ]

    net.load_state_dict(torch.load(files[0]))
    for i, f in enumerate(files[1:]):
        net2.load_state_dict(torch.load(f))
        moving_average(net, net2, 1. / (i + 2))

    img_size = 768
    batch_size = 6


    train_dataloader =  DataLoader(
        dataset=torch_data_load.AirbusDS(mode='train',img_size=img_size,transform=None),
        shuffle=True,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=np.random.seed(SEED)
    )



    net.cuda()
    bn_update(train_dataloader, net)

    torch.save(net.state_dict(), swa_path+'/swa')