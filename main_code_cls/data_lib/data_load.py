import numpy as np
from torch.utils import data
import pandas as pd
import os
import glob
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import ToTensor, Normalize, Compose
import torch
import albumentations as al



class AIRBUS_CLASS(data.Dataset):

    def __init__(self,mode,img_size=768,transform=None,if_name = False):

        assert mode in ['train','val','valall']


        self.mode = mode

        self.img_size = img_size
        self.if_name = if_name
        self.root_path = '/mnt/sda1/don/documents/airbus/data/v2/'
        self.imgpath = self.root_path+'train_v2/'
        self.transform = transform
        self.base_tranform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ]
        )




        if self.mode == 'train':
            self.is_label = pd.read_csv('/mnt/sda1/don/documents/airbus/data/v2/cls_split_list/is_ship_train.csv')
            self.img_ids = self.is_label['ImageId'].values
            self.img_label = self.is_label['is_ship'].values
        elif self.mode == 'valall':
            self.is_label = pd.read_csv('/mnt/sda1/don/documents/airbus/data/v2/cls_split_list/is_ship_val.csv')
            self.img_ids = self.is_label['ImageId'].values
            self.img_label = self.is_label['is_ship'].values
        else:
            self.is_label = pd.read_csv('/mnt/sda1/don/documents/airbus/data/v2/cls_split_list/is_ship_inp_val.csv')
            self.img_ids = self.is_label['ImageId'].values
            self.img_label = self.is_label['is_ship'].values



        self.len = len(self.img_ids)





    def __getitem__(self, index):

        image = cv2.imread(str(self.imgpath + self.img_ids[index])).astype(np.float32)

        if self.img_size!=768:
            image = cv2.resize(image, (self.img_size, self.img_size))

        img_id = self.img_ids[index]
        is_ship = np.array(self.img_label[index])


        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']



        if self.if_name:
            return self.base_tranform(image), torch.from_numpy(is_ship).float(),img_id
        else:
            return self.base_tranform(image), torch.from_numpy(is_ship).float()


    def __len__(self):
        return self.len





class AIRBUS_CLASS_TEST(data.Dataset):
    """
    A customized data loader.
    """
    # 83256 20814
    def __init__(self,img_size=768,if_name = False):
        """ Intialize the dataset
        """



        self.img_size = img_size
        self.if_name = if_name
        self.root_path = '/mnt/sda1/don/documents/airbus/data/v2'
        self.base_tranform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            ]
        )




        self.img_ids = os.listdir(self.root_path+'test_v2')


        self.len = len(self.img_ids)





    def __getitem__(self, index):
        """ Get a sample from the dataset
        """

        image = cv2.imread(str(self.root_path + 'test_v2/' + self.img_ids[index])).astype(np.float32)

        if self.img_size!=768:
            image = cv2.resize(image, (self.img_size, self.img_size)).astype(np.float32)


        if self.if_name:
            return self.base_tranform(image),str(self.img_ids[index])
        else:
            return self.base_tranform(image)


    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


#
if __name__ == '__main__':

    train_transform = al.Compose(
        [
            al.VerticalFlip(p=0.5),
            al.HorizontalFlip(p=0.5),
            al.RandomRotate90(p=0.5)
        ]
    )

    airimg = AIRBUS_CLASS('val', transform=train_transform,if_name=True)
    # Use the torch dataloader to iterate through the dataset
    loader = data.DataLoader(airimg, batch_size=1, shuffle=False, num_workers=4)

    # get some images
    dataiter = iter(loader)

    for x in range(100):
        images,is_salt,names = next(dataiter)
        images = images.numpy()
        #print(is_salt.shape)
        print(is_salt[0])

        mask = np.transpose(images[0,:,:,:],(1,2,0)).astype(np.uint8)
        #plt.figure(figsize=(15,15))
        #plt.subplot(211)
        plt.imshow(mask)
        #plt.subplot(212)
        #plt.imshow(masks[0,0,:,:],alpha=0.5)
        plt.show()






