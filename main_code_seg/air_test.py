import os

from collections import OrderedDict
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm
import click
import numpy as np
import torch
import logging
import sys
from main_code_seg.data_lib import torch_data_load
from main_code_seg.data_lib.torch_agument import *
from main_code_seg.utils import util
import time
from torch.nn import functional as F

from main_code_seg.model_include import *
import random
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import scipy.misc as misc
import pandas as pd
from skimage.morphology import binary_opening, disk,label
from main_code_seg.utils import util
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
cudnn.deterministic = True
cudnn.benchmark = True


def flip_n(image,mask=None):
    if mask is not None:
        return image,mask
    else:
        return image

def flip_v(image,mask=None):
    if mask is not None:
        return image.flip(2),mask.flip(2)
    else:
        return image.flip(2)


def flip_h(image,mask=None):
    if mask is not None:
        return image.flip(3),mask.flip(3)
    else:
        return image.flip(3)

def flip_vh(image, mask=None):
    if mask is not None:
        return image.flip(2).flip(3), mask.flip(2).flip(3)
    else:
        return image.flip(2).flip(3)

def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:

        net.load_state_dict(torch.load(snapshot))

    net = net.cuda()
    return net, epoch

def make_loader(img_size,batch_size,shuffle=False):
    return DataLoader(
        dataset=torch_data_load.AirbusDS_TEST(img_size=img_size),
        shuffle=shuffle,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=np.random.seed(SEED)
    )


def plain_split(mask):
    mask = binary_opening(mask > 0.5, disk(2))
    labels = label(mask)
    has_unique = np.unique(labels[labels > 0])
    all_masks = []
    if len(has_unique)>0:
        for idx,k in enumerate(has_unique):
            all_masks.append((labels==k)*1.0)

    return all_masks


def m_test(weight,file_name):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    ex_name = 'dense_normal'
    model_name = 'dense_normal'
    batch_size = 24
    TTA = True
    CCF = True
    IMAGE_SIZE = 768

    TTA_list = {
        'n':flip_n,
        'v':flip_v,
        'h':flip_h,
        'vh':flip_vh,
    }
    valid_loader = make_loader(img_size=IMAGE_SIZE,
                               batch_size=batch_size,
                               shuffle=False)


    init_path = weight




    net, starting_epoch = build_network(init_path, model_name)


    with torch.no_grad():

        net.eval()
        out_pred_rows = []
        for batch_num, (inputs, paths) in enumerate(tqdm(valid_loader, desc='Predict')):
            if not TTA:

                inputs = Variable(inputs).cuda()

                outputs = net(inputs)

                outputs = torch.sigmoid(outputs)
                outputs = torch.squeeze(outputs.data.cpu(), dim=1)
                outputs = outputs.numpy()
            else:
                outputs = []

                for tta_type in TTA_list.keys():
                    vxt = TTA_list[tta_type](inputs)
                    vxv = Variable(vxt).cuda()

                    fuse_logitv = net(vxv)



                    fuse_logitv = TTA_list[tta_type](fuse_logitv)
                    out_toiou = F.sigmoid(fuse_logitv)
                    out_toiou = torch.squeeze(out_toiou.data.cpu(), dim=1)
                    out_toiou = out_toiou.numpy()

                    outputs.append(out_toiou)
                outputs = np.stack(outputs,axis=-1)
                outputs = np.mean(outputs,axis=-1)



            for i, image_name in enumerate(paths):

                mask = outputs[i,:,:]

                cur_seg = np.where(mask > 0.45, 1, 0)
                cur_rles = util.multi_rle_encode(cur_seg)
                if len(cur_rles) > 0:
                    for c_rle in cur_rles:
                        out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': c_rle}]
                else:
                    out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]

            # if batch_num>20:
            #     break
        submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
        submission_df.to_csv('submission_c/{}.csv'.format(file_name), index=False)

        if CCF:
            boat_df = pd.read_csv('new_score.csv')
            is_boat = boat_df.score > 0.5
            has_boat_list = list(boat_df[is_boat]['ImageId'])

            out_pred_rows = []

            for i in tqdm(range(len(submission_df))):
                id = submission_df['ImageId'][i]
                rle = submission_df['EncodedPixels'][i]

                if id in has_boat_list:
                    if not isinstance(rle,str):
                        out_pred_rows += [{'ImageId': id, 'EncodedPixels': None}]
                    else:
                        out_pred_rows += [{'ImageId': id, 'EncodedPixels': rle}]
                else:
                    out_pred_rows += [{'ImageId': id, 'EncodedPixels': None}]

            submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
            submission_df.to_csv('submission_c/{}_ns5.csv'.format(file_name), index=False)




if __name__ == '__main__':
    #time.sleep(60*20)
    m_test(weight='/mnt/sda1/don/documents/airbus/main_code_seg/experiments/swa/smixa0a1/swa',
           file_name='swa')


