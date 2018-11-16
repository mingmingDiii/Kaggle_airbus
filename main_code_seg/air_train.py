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
from torch.optim.lr_scheduler import MultiStepLR
import logging
import sys
from main_code_seg.data_lib import torch_data_load
#from main_code_seg.data_lib.torch_agument import *
import albumentations as al
from main_code_seg.utils import util

from torch.nn import functional as F
#from main_code_seg.data_lib.tgs_agument import *
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
import time
import datetime
from main_code_seg.model_include import *
import random
import torch.backends.cudnn as cudnn
from main_code_seg.losses import base_loss
from skimage.morphology import binary_opening, disk,label
import scipy.misc as misc
import cv2

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
cudnn.deterministic = True
cudnn.benchmark = True


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        epoch = os.path.basename(snapshot).split('_')[-1]
        epoch = int(epoch)+1
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))

    net = net.cuda()
    return net, epoch

def make_loader(mode,batch_size,img_size,shuffle=False,transform=None):
    return DataLoader(
        dataset=torch_data_load.AirbusDS(mode=mode,img_size=img_size,transform=transform),
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


train_transform = al.Compose(
    [al.VerticalFlip(p=0.5),
     al.HorizontalFlip(p=0.5),
     al.RandomRotate90(p=0.5),
     al.ShiftScaleRotate(p=0.5,shift_limit=0.1, scale_limit=0.1, rotate_limit=45)
     ]
)

train_transform_crop = al.Compose(
    [al.VerticalFlip(p=0.5),
     al.HorizontalFlip(p=0.5),
     al.RandomRotate90(p=0.5),
     al.ShiftScaleRotate(p=0.5,shift_limit=0.1, scale_limit=0.1, rotate_limit=45),
    #al.RandomCrop(p=1,height=384,width=384)
     ]
)



def mtrain():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    ex_name = 'dn_SLovDiB768_has_norm'
    model_name = 'dense_normal'
    describe = " "
    batch_size = 8
    epochs = 30

    IMAGE_SIZE = 768
    # lr_decay_step = 1000000
    # early_stop_step = 10000000
    # lr_decay_factor = 1

    opt = 'adam'

    AVE = [1,2,3,4]

    ##################################################### setting experiments dir ###################################################
    exp_dir = 'experiments/{}/'.format(ex_name)


    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)





    train_loader = make_loader(mode='train',
                               batch_size=batch_size,
                               shuffle=True,
                               img_size = IMAGE_SIZE,
                               transform=train_transform_crop)
    valid_loader = make_loader(mode='val',
                               batch_size=batch_size,
                               shuffle=False,
                               img_size = IMAGE_SIZE,
                               transform=None)

    for a in AVE:

        lr = 1e-4
        val_loss = 0.0
        val_miou = 0.0
        # val_loss_best = 100000.
        # val_miou_best = 0.0
        # train_loss_best = 100000.
        # train_miou_best = 0.0
        #
        # best_valloss_epoch = 0
        # best_trainloss_epoch = 0
        # early_rec = 0
        # lr_rec = 0

        ######  make ave path #######
        ave_path = exp_dir+'a{}/'.format(a)
        if not os.path.exists(ave_path):
            os.mkdir(ave_path)

        ### weights path
        weights_path = ave_path + 'weights/'
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)

        ### board path
        board_path = ave_path + 'board/'
        if not os.path.exists(board_path):
            os.mkdir(board_path)
        writer = SummaryWriter(board_path)
        ################# config file ###########################################
        # config_path = ave_path + 'config/'
        # if not os.path.exists(config_path):
        #     os.mkdir(config_path)



        config_file = open(ave_path+'config.txt','a')
        config_file.write('Time:\t{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        config_file.write('ex_name:\t{}\n'.format(ex_name))
        config_file.write('model_name:\t{}\n'.format(model_name))
        config_file.write('batch_size:\t{}\n'.format(batch_size))
        config_file.write('image_size:\t{}\n'.format(IMAGE_SIZE))

        config_file.write('opt:\t{}\n'.format(opt))
        config_file.write('init_lr:\t{}\n'.format(lr))
        # config_file.write('lr_decay_step:\t{}\n'.format(lr_decay_step))
        # config_file.write('early_stop_step:\t{}\n'.format(early_stop_step))
        # config_file.write('lr_decay_factor:\t{}\n'.format(lr_decay_factor))


        config_file.write('describe:\t{}\n'.format(describe))



        if a==0:
            init_path = None
        else:

            # w_names = os.listdir(exp_dir + 'a{}/weights/'.format(a-1))
            # epochst = [os.path.basename(snapshot).split('_')[1] for snapshot in w_names]
            # epochst = np.array(epochst, dtype=np.int32)
            # best_epoch = np.max(epochst)
            # init_path = exp_dir + 'a{}/weights/busnet_{}'.format(a-1, best_epoch)
            # print(init_path)
            init_path = '/mnt/sda1/don/documents/airbus/main_code_seg/experiments/dn_SLovDiB768_has_norm/a1/weights/busnet_5'
        net, st = build_network(init_path, model_name)



        opt_list = {
            'adam': lambda x: optim.Adam(net.parameters(),lr=x),
            'sgd':lambda x: optim.SGD(net.parameters(),lr=x,momentum=0.9, weight_decay=0.0001), # momentum=0.9, weight_decay=0.0001
            'rms':lambda x:optim.RMSprop(net.parameters(),lr=x)
        }
        cri_dict ={
            'bce_logits':nn.BCEWithLogitsLoss(),
            'jac':base_loss.LossBinary(jaccard_weight=5),

            'lov': base_loss.LovLoss(),
            'slov':base_loss.SLovLoss(),
            'mix_dice_bce':base_loss.Mix_softDice_bce(),
            'mix_dice_bce_lov':base_loss.Mix_softDice_bce_lov(),
            'mix_dice_bce_slov':base_loss.Mix_softDice_bce_slov()
        }


        #scheduler = MultiStepLR(optimizer, gamma=0.1,milestones=[50,100,150])

        cri = 'mix_dice_bce_slov'
        for epoch in range(st,st+epochs):

            # if epoch>=5:
            #     cri='mix_dice_bce_slov'

            if epoch>=10:
                lr = 1e-5
            if epoch>=30:
                lr = 1e-6
            # if epoch>20:
            #     lr = 1e-6

            optimizer = opt_list[opt](lr)
            seg_criterion = cri_dict[cri]

            epoch_losses = []



            net.train()
            #net.set_mode(mode='train',is_freeze_bn=False)

            train_iterator = tqdm(train_loader, total=len(train_loader))

            #t=0
            for x, y in train_iterator:
                # t+=1
                # if t>20:
                #     break
                optimizer.zero_grad()

                x = Variable(x).cuda()
                y = Variable(y).cuda()


                y_pred = net(x)
                loss = seg_criterion(y_pred, y)


                epoch_losses.append(loss.data)




                status = "[{}][{:03d}]" \
                         "all = {:0.5f}," \
                         "LR = {:0.7f}, " \
                         "vall = {:0.5f}, vmiou = {:0.5f}".format(
                    ex_name, epoch,
                    np.mean(epoch_losses),
                    lr,
                    val_loss, val_miou)
                train_iterator.set_description(status)

                loss.backward()
                optimizer.step()


            #scheduler.step()

            train_loss = np.mean(epoch_losses)
            torch.save(net.state_dict(), os.path.join(weights_path, '_'.join(["busnet", str(epoch)])))
            # if train_loss<train_loss_best:
            #     train_loss_best = train_loss
            #     best_trainloss_epoch = epoch

            with torch.no_grad():
                # make val
                net.eval()
                # net.set_mode(mode='eval', is_freeze_bn=True)
                val_losses = []
                val_mious = []
                val_iterator = valid_loader


                for vx, vy in val_iterator:

                    vxv = Variable(vx).cuda()
                    if cri == 'lov':
                        vyv = vy.cuda()
                    else:
                        vyv = Variable(vy).cuda()


                    fuse_logitv = net(vxv)

                    seg_loss_fusev = seg_criterion(fuse_logitv, vyv)


                    val_losses.append(seg_loss_fusev.data)

                    out_toiou = F.sigmoid(fuse_logitv)
                    out_toiou = torch.squeeze(out_toiou.data.cpu(), dim=1).numpy()

                    vy_toiou = np.squeeze(vy.numpy(),axis=1)

                    #if out_toiou.shape[-1]!=768:


                    for b in range(out_toiou.shape[0]):
                        # if vy_toiou.shape[-1]!=768:
                        #     y_pred_sp = cv2.resize(out_toiou[b],(768,768))
                        #     y_sp = cv2.resize(vy_toiou[b], (768, 768))
                        # else:
                        y_pred_sp = out_toiou[b]
                        y_sp = vy_toiou[b]
                        y_pred_sp = plain_split(y_pred_sp)
                        y_sp = plain_split(y_sp)

                        iou = util.f2(y_pred_sp, y_sp)

                        val_mious.append(iou)
            val_loss = np.mean(val_losses)
            val_miou = np.mean(val_mious)

            writer.add_scalar('epoch_train_loss', train_loss, epoch)
            writer.add_scalar('epoch_val_loss', val_loss, epoch)
            writer.add_scalar('epoch_val_miou', val_miou, epoch)
            writer.add_scalar('lr', lr, epoch)


            writer.add_scalars('epoch_group',{
                'train_loss':train_loss,
                'val_loss':val_loss,
                'val_miou':val_miou,
                'lr':lr
            },epoch)

            config_file.write('Epoch : {}, Val_iou : {}\n'.format(epoch,val_miou))

        config_file.close()



if __name__ == '__main__':
    #time.sleep(60*30)
    mtrain()


