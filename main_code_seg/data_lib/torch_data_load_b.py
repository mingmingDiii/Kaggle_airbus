from main_code_seg.utils import util
import numpy as np
from torch.utils import data
import pandas as pd
import os
import glob
import torchvision
import torchvision.transforms as transforms
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
import albumentations as al
from skimage.morphology import dilation, watershed, square, erosion
from skimage import measure

# from albumentations import (
# HorizontalFlip,
# VerticalFlip,
# RandomRotate90,
# OneOf,
# Compose
# )
def return_img(img):
    img = np.transpose(img,(1,2,0))
    img[:,:,0] = img[:,:,0]*0.229+0.485
    img[:,:,1] = img[:,:,1]*0.224+0.456
    img[:,:,2] = img[:,:,2]*0.225+0.406

    return (img*255).astype(np.uint8)




class AirbusDS(data.Dataset):
    """
    A customized data loader.
    """
    # 83256 20814
    def __init__(self, mode='train',img_size =384,transform=None):
        """ Intialize the dataset
        """
        assert mode in ['train','val','valall']


        self.mode = mode
        self.root_path = '/mnt/sda1/don/documents/airbus/data/v2/'
        self.imgpath = self.root_path+'train_v2/'
        self.transform = transform
        self.img_size = img_size
        self.img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.masks = pd.read_csv(self.root_path+'train_ship_segmentations_v2.csv').fillna(-1)
        if self.mode=='train':
            self.img_ids = open(self.root_path+'split_list/has_train.csv').read().splitlines()
        elif self.mode=='val':
            self.img_ids = open(self.root_path + 'split_list/inp_val.csv').read().splitlines()
        elif self.mode=='valall':
            self.img_ids = open(self.root_path+'split_list/val.csv').read().splitlines()


        self.len = len(self.img_ids)


    def __getitem__(self, index):
        """ Get a sample from the dataset
        """

        image = cv2.imread(str(self.imgpath+self.img_ids[index]))#.astype(np.float32)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        masks = self.get_mask(self.img_ids[index])

        if self.img_size!=768:

            image = cv2.resize(image,dsize=(self.img_size,self.img_size))
            masks = cv2.resize(masks,dsize=(self.img_size,self.img_size))
            masks = (masks > 0.5).astype(np.float32)

            #print(np.max(mask))


        #image,mask = self._transform(image,mask)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image']
            masks = augmented['mask']

        masks = np.transpose(masks,(2,0,1))
        if self.mode=='valall':
            return self.img_transform(image), torch.from_numpy(masks).float(),str(self.img_ids[index])
        else:
            return self.img_transform(image), torch.from_numpy(masks).float()#self.img_transform(image), torch.from_numpy(mask).float()


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

    def get_mask(self, ImageId):

        img_masks = self.masks.loc[self.masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        # Take the individual ship masks and create a single mask array for all ships
        all_labels = np.zeros((768, 768))
        if isinstance(img_masks[0],str):
            for i,mask in enumerate(img_masks):
                all_labels += util.rle_decode(mask)*(i+1)

            all_labels = all_labels.astype(np.int8)
            masks = self.create_mask(all_labels)
        else:
            masks = np.zeros((768,768,3),np.uint8)
        #print(all_masks)
        return masks

    def create_mask(self,labels):
        # labels = measure.label(labels, neighbors=8, background=0)
        tmp1 = dilation(labels > 0, square(9))

        tmp2 = watershed(tmp1, labels, mask=tmp1, watershed_line=True) > 0

        tmp = tmp1 ^ tmp2
        tmp = dilation(tmp, square(7))

        # msk = (255 * tmp).astype('uint8')

        props = measure.regionprops(labels)
        msk0 = 1 * (labels > 0)
        msk0 = msk0.astype('uint8')
        # plt.imshow(msk0)
        # plt.show()

        msk1 = np.zeros_like(labels, dtype='bool')

        max_area = np.max([p.area for p in props])

        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                if labels[y0, x0] == 0:
                    if max_area > 4000:
                        sz = 6
                    else:
                        sz = 3
                else:
                    sz = 3
                    if props[labels[y0, x0] - 1].area < 300:
                        sz = 1
                    elif props[labels[y0, x0] - 1].area < 2000:
                        sz = 2
                uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                                 max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
                if len(uniq[uniq > 0]) > 1:
                    msk1[y0, x0] = True
                    msk0[y0, x0] = 0

        msk1 = 1 * msk1
        msk1 = msk1.astype('uint8')

        msk_all = 1 * (labels > 0)
        msk_all = msk_all.astype('uint8')

        msk = np.stack((msk_all, msk1, msk0), axis=-1)  # mask_all,board,mask-board

        return msk


class AirbusDS_TEST(data.Dataset):
    """
    A customized data loader.
    """
    # 83256 20814
    def __init__(self,img_size,transform=None,aug=False):
        """ Intialize the dataset
        """

        self.root_path = '/mnt/sda1/don/documents/airbus/data/v2/'
        self.imgpath = self.root_path+'test_v2/'
        self.transform = transform
        self.img_size = img_size
        self.img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.img_ids = os.listdir('/mnt/sda1/don/documents/airbus/data/v2/test_v2')
        self.len = len(self.img_ids)



    def __getitem__(self, index):
        """ Get a sample from the dataset
        """


        image = cv2.imread(str(self.imgpath+self.img_ids[index]))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if self.img_size!=768:

            image = cv2.resize(image,dsize=(self.img_size,self.img_size))

        return self.img_transform(image),str(self.img_ids[index])


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

#
if __name__ == '__main__':

    # train_transform = DualCompose([
    #     #HorizontalFlip(0.5),
    #     #VerticalFlip(1.0),
    #     Rotate(45,1.0)
    #     #RandomCrop((256, 256, 3)),
    #     # ImageOnly(RandomBrightness()),
    #     # ImageOnly(RandomContrast()),
    # ])
    train_transform = al.Compose(
        [al.VerticalFlip(p=0.5),
         al.HorizontalFlip(p=0.5),
         al.RandomRotate90(p=0.5),
         al.ShiftScaleRotate(p=1,shift_limit=0., scale_limit=0., rotate_limit=45)]
    )

    airimg = AirbusDS('val', img_size=384,transform=train_transform)
    # Use the torch dataloader to iterate through the dataset
    loader = data.DataLoader(airimg, batch_size=1, shuffle=False, num_workers=4)

    # get some images
    dataiter = iter(loader)

    for x in range(100):
        images,masks = next(dataiter)
        #images = images.flip(2)
        images = images.numpy()
        masks = masks.numpy()
        print(np.max(masks))
        print(np.min(masks))
        #images = images.numpy()

        images = return_img(images[0])
        plt.figure(figsize=(15,15))

        plt.subplot(311)
        plt.imshow(images)
        plt.imshow(masks[0,0,:,:],alpha=0.5)

        plt.subplot(312)
        plt.imshow(images)
        plt.imshow(masks[0,1,:,:],alpha=0.5)

        plt.subplot(313)
        plt.imshow(images)
        plt.imshow(masks[0,2,:,:],alpha=0.5)

        plt.show()






