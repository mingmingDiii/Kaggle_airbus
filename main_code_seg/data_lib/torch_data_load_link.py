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
from scipy.ndimage.morphology import distance_transform_edt
from skimage import filters
import albumentations as al

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

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h

def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),        (x - 1, y),                 (x + 1, y),         (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]

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
        self.nei = 8
        self.masks = pd.read_csv(self.root_path+'train_ship_segmentations_v2.csv').fillna(-1)
        if self.mode=='train':
            self.img_ids = open(self.root_path+'split_list/has_train.csv').read().splitlines()
        elif self.mode=='val':
            self.img_ids = open(self.root_path + 'split_list/inp_val.csv').read().splitlines()
        elif self.mode=='valall':
            self.img_ids = open(self.root_path+'split_list/val.csv').read().splitlines()


        self.len = len(self.img_ids)


    def get_mask(self, ImageId):

        img_masks = self.masks.loc[self.masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((self.img_size, self.img_size))

        link_label = np.zeros((self.img_size,self.img_size,self.nei))
        if img_masks == [-1]:
            return all_masks.astype(np.float32),link_label.astype(np.float32)
        for mask in img_masks:
            decode_mask = util.rle_decode(mask)

            if self.img_size!=768:
                decode_mask = cv2.resize(decode_mask,(self.img_size,self.img_size))
                decode_mask = (decode_mask > 0.5).astype(np.float32)


            edge = filters.roberts(decode_mask)
            edge = np.where(edge > 0, 1.0, 0.0)


            border_cords = np.where(edge)
            border_points = zip(*border_cords)

            def in_bbox(nx, ny):
                return decode_mask[ny, nx]

            for y, x in border_points:
                if self.nei==4:
                    neighbours = get_neighbours_4(x, y)
                else:
                    neighbours = get_neighbours_8(x, y)
                for n_idx, (nx, ny) in enumerate(neighbours):
                    if not is_valid_cord(nx, ny, self.img_size, self.img_size) or not in_bbox(nx, ny):
                        link_label[y, x, n_idx] = 1

            all_masks += decode_mask

        # plt.figure(figsize=(15, 15))
        # plt.imshow(all_masks)
        # plt.show()
        # for i in range(self.nei):
        #     plt.figure(figsize=(15, 15))
        #     plt.imshow(link_label[:, :, i])
        #     plt.show()

        #print(all_masks)
        return all_masks.astype(np.float32),link_label.astype(np.float32)



    def __getitem__(self, index):
        """ Get a sample from the dataset
        """

        image = cv2.imread(str(self.imgpath+self.img_ids[index]))#.astype(np.float32)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask, link_label = self.get_mask(self.img_ids[index])

        if self.img_size!=768:

            image = cv2.resize(image,dsize=(self.img_size,self.img_size))

        mask = mask[:,:,np.newaxis]
        mask_edge = np.concatenate([mask, link_label], axis=-1)

        if self.transform is not None:

            augmented2 = self.transform(image=image, mask=mask_edge)
            image = augmented2['image']
            mask_edge = augmented2['mask']


        mask_edge = np.transpose(mask_edge,(2,0,1))
        if self.mode=='valall':
            return self.img_transform(image), torch.from_numpy(mask_edge).float(),str(self.img_ids[index])
        else:
            return self.img_transform(image), torch.from_numpy(mask_edge).float()#self.img_transform(image), torch.from_numpy(mask).float()


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

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
    loader = data.DataLoader(airimg, batch_size=1, shuffle=False, num_workers=1)

    # get some images
    dataiter = iter(loader)

    for x in range(100):
        images,masks = next(dataiter)
        #images = images.flip(2)
        images = images.numpy()
        masks = masks.numpy()
        image = return_img(images[0])

        plt.figure(figsize=(15,15))

        plt.subplot(211)
        plt.imshow(image)
        plt.imshow(masks[0,0,:,:],alpha=0.5)
        plt.subplot(212)
        plt.imshow(masks[0,-1,:,:])


        plt.show()






