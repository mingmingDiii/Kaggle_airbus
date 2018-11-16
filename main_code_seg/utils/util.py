import numpy as np
import torch
import json
from datetime import datetime
from torch.autograd import Variable
from skimage.morphology import label
import cv2
import matplotlib.pyplot as plt


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def multi_rle_encode(img):
    labels = label(img)
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def mask_overlay(image, mask, color=(0, 1, 0)):
    """
    Helper function to visualize mask on the top of the image
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img

def imshow(img, mask, title=None):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    mask = mask.numpy().transpose((1, 2, 0))
    mask = np.clip(mask, 0, 1)
    fig = plt.figure(figsize = (6,6))
    plt.imshow(mask_overlay(img, mask))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)





################ AIRBUS METRIC ###################

def iou(img_pred,img_true):
    i = np.sum((img_true*img_pred)>0)
    u = np.sum((img_true+img_pred)>0)+1e-20
    return i/u


def f2(masks_pred,masks_true):

    thresholds = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

    if len(masks_pred)==len(masks_true)==0:
        return 1.0

    f2_toal = 0

    for t in thresholds:
        tp,fp,fn = 0,0,0
        ious = {}
        for i,mt in enumerate(masks_true):
            found_match = False
            for j,mp in enumerate(masks_pred):
                miou = iou(mp,mt)
                ious[100*i+j] = miou
                if miou>=t:
                    found_match = True
            if not found_match:
                fn+=1
        for j,mp in enumerate(masks_pred):
            found_match = False
            for i,mt in enumerate(masks_true):
                miou = ious[100*i+j]
                if miou>=t:
                    found_match=True
                    break
            if found_match:
                tp +=1
            else:
                fp +=1
        f2 = (5*tp)/(5*tp+4*fn+fp)
        f2_toal += f2

    return f2_toal/len(thresholds)


















