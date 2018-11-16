import os
import numpy as np

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch
import logging

from main_code_cls.models import densenet
from main_code_cls.models import resnet
from main_code_cls.data_lib import data_load

from torch.nn import functional as F
import albumentations as al

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
models = {

    'densenet121': lambda :densenet.densenet121(pretrained=True,num_classes=1),
    'resnet101': lambda : resnet.resnet101(pretrained=True, num_classes=1)
}


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

def make_loader(mode,img_size,batch_size,shuffle=False,transform=None):
    return DataLoader(
        dataset=data_load.AIRBUS_CLASS(mode=mode,img_size=img_size, transform=transform),
        shuffle=shuffle,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

def fast_hist(a, b, n):
    """
    Fast 2D histogram by linearizing.
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def acc_batch(target,output,numclass=2):
    hist = fast_hist(np.int64(target), np.int64(output), numclass)
    acc = np.diag(hist).sum() / hist.sum()

    return acc

train_transform = al.Compose(
    [
        al.VerticalFlip(p=0.5),
        al.HorizontalFlip(p=0.5),
        al.RandomRotate90(p=0.5)
    ]
)


batch_size = 16
epochs = 20
lr = 1e-5
IMAGE_SIZE = 384


ex_name = 'densenet121'
model_name = 'densenet121'
val_loss = 0.0
val_miou = 0.0
val_loss_best = 100000.
val_miou_best = 0.0
early_rec = 0
lr_rec = 0
lr_decay_step = 10000000
early_stop_step = 25000000

ex_path = 'experiments/{}/'.format(ex_name)
if not os.path.exists(ex_path):
    os.mkdir(ex_path)

weights_path = ex_path+'weights/'
if not os.path.exists(weights_path):
    os.mkdir(weights_path)
# weights_path = ex_path+'weights/'
# if not os.path.exists(weights_path):
#     os.mkdir(weights_path)

config_file = open(ex_path+'config.txt','a')

train_loader = make_loader(mode='train', img_size=IMAGE_SIZE,batch_size=batch_size, shuffle=True, transform=train_transform)
valid_loader = make_loader(mode='val', img_size=IMAGE_SIZE,batch_size=batch_size, transform=None)

init_path ='/home/don/documents/airbus/main_code_cls/experiments/densenet121/weights/airbus_cls_18'
net, starting_epoch = build_network(init_path, model_name)

#scheduler = MultiStepLR(optimizer, milestones=[20,40],gamma=0.1)



cri_dict ={

    'bce_logits':nn.BCEWithLogitsLoss()
}
cri = 'bce_logits'

for epoch in range(starting_epoch, starting_epoch + epochs):

    optimizer = optim.Adam(net.parameters(), lr=lr)


    seg_criterion = cri_dict[cri]


    epoch_losses = []

    net.train()

    train_iterator = tqdm(train_loader, total=len(train_loader))


    #t=0
    for x, y in train_iterator:
        # t+=1
        # if t>5:
        #     break
        optimizer.zero_grad()

        x = Variable(x).cuda()
        out = net(x)

        y = Variable(y).cuda()

        loss = seg_criterion(out, y)

        epoch_losses.append(loss.data)
        status = "[{:03d}] loss = {:0.5f} avg = {:0.5f}, LR = {:0.7f} Last val_loss = {:0.5f} Last val_miou = {:0.5f}".format(
            epoch, loss.data, np.mean(epoch_losses), lr, val_loss,val_miou)
        train_iterator.set_description(status)

        loss.backward()
        optimizer.step()
    #scheduler.step()

    train_loss = np.mean(epoch_losses)

    with  torch.no_grad():
        #make val
        net.eval()
        val_losses = []
        val_mious = []
        val_iterator = valid_loader
        for vx,vy in val_iterator:
            vxv = Variable(vx).cuda()
            out = net(vxv)


            vyv = Variable(vy).cuda()
            loss_valt = seg_criterion(out, vyv)
            val_losses.append(loss_valt.data)


            out_toiou = torch.sigmoid(out)
            #out_toiou = torch.squeeze(out_toiou.data.cpu(),dim=1)

            out_toiou = out_toiou.data.cpu().numpy()
            #print(out_toiou)
            out_toiou = np.where(out_toiou>0.5,1,0)
            #print(out_toiou)
            vy_toiou = vy.numpy()

            miou = acc_batch(target=vy_toiou,output=out_toiou,numclass=2)#util.iou_metric_batch(y_pred_in=out_toiou,y_true_in=vy_toiou)

            val_mious.append(miou)
        val_loss = np.mean(val_losses)
        val_miou = np.mean(val_mious)


    torch.save(net.state_dict(), os.path.join(weights_path, '_'.join(["airbus_cls", str(epoch)])))
    config_file.write('Epoch : {},  ACC : {} \n'.format(epoch, val_miou))

    # Save best and early stopping
    early_rec += 1
    lr_rec +=1
    if val_miou>val_miou_best:
        val_miou_best = val_miou
        # torch.save(net.state_dict(), os.path.join(weights_path, '_'.join(["airbus_cls", str(epoch + 1)])))
        # config_file.write('Epoch : {},  ACC : {}'.format(epoch+1,val_miou_best))
        early_rec = 0
        lr_rec = 0

    if lr_rec>lr_decay_step:
        print('decay lr',epoch)
        lr = lr *0.5
        lr = max(1e-6,lr)
        lr_rec=0

    if early_rec>early_stop_step:
        print('Early stop',epoch)
        print('best val miou',val_miou_best)
        #torch.save(net.state_dict(), os.path.join(weights_path, '_'.join(["airbus_cls", str(epoch + 1),'_',str(val_miou_best)])))
        break









