import os
import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
models = {

    'densenet121': lambda :densenet.densenet121(pretrained=True,num_classes=1),
    'resnet101': lambda : resnet.resnet101(pretrained=True, num_classes=1)
}

def return_img(img):
    mean = [.485, .456, .406]
    std = [.229, .224, .225]
    img = np.transpose(img,(1,2,0))

    img[:, :, 0] = img[:, :, 0] * 0.229 + 0.485
    img[:, :, 1] = img[:, :, 1] * 0.224 + 0.456
    img[:, :, 2] = img[:, :, 2] * 0.225 + 0.406
    return (img*255).astype(np.uint8)

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

def make_loader(img_size,batch_size,shuffle=False):
    return DataLoader(
        dataset=data_load.AIRBUS_CLASS_TEST(img_size=img_size,if_name=True),
        shuffle=shuffle,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )


batch_size = 16

IMAGE_SIZE = 384


ex_name = 'densenet121'
model_name = 'densenet121'


trans_df = pd.read_csv('/home/don/documents/airbus/data/v2/sample_submission_v2.csv')
test_loader = make_loader(img_size=IMAGE_SIZE,batch_size=batch_size)

init_path ='/home/don/documents/airbus/main_code_cls/experiments/densenet121/weights/airbus_cls_26'
net, starting_epoch = build_network(init_path, model_name)



with  torch.no_grad():
    #make val
    net.eval()

    test_loader = tqdm(test_loader,total=len(test_loader))
    all_scores = dict()

    for vx,names in test_loader:

        vxv = Variable(vx).cuda()
        out = net(vxv)


        out_toiou = torch.sigmoid(out)

        out_toiou = out_toiou.data.cpu().numpy()


        # img = vx.numpy()[0]
        #
        #
        # print(out_toiou)
        #
        # plt.imshow(np.uint8(np.transpose(img,(1,2,0))))
        # plt.show()

        for c_id,c_score in zip(names,out_toiou):
            all_scores[c_id] = c_score

    trans_df['score'] = trans_df['ImageId'].map(lambda x:all_scores.get(x,0))
    trans_df[['ImageId','score']].to_csv('new_score.csv',index=False)






