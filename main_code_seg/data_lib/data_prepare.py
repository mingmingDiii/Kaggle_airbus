import pandas as pd
import os
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
def prepare1():

    ff = open('../../data/train_ids.csv').read().splitlines()
    masks = pd.read_csv(os.path.join('../../data/',
                                     'train_ship_segmentations.csv'))
    # print(masks.shape[0], 'masks found')
    # print(masks['ImageId'].value_counts().shape[0])

    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
    unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

    #masks.drop(['ships'], axis=1, inplace=True)
    print(unique_img_ids.sample(100))
    print(len(unique_img_ids))
    # print(masks.columns)


    train_ids, valid_ids = train_test_split(unique_img_ids,
                     test_size = 0.2,
                     stratify = unique_img_ids['ships'], random_state=1337)
    print(len(train_ids))
    print(len(valid_ids))
    train_ids['ImageId'].to_csv('../../data/train_ids.csv',index=False)
    valid_ids['ImageId'].to_csv('../../data/val_ids.csv',index=False)
    unique_img_ids['ImageId'].to_csv('../../data/trainval_ids.csv',index=False)
    # train_df = pd.merge(masks, train_ids)
    # valid_df = pd.merge(masks, valid_ids)
    # print(train_df.shape[0], 'training masks')
    # print(valid_df.shape[0], 'validation maskself.imgpath+self.img_ids[index]s')


def prepare_balance():
    masks = pd.read_csv('../../data/train_ship_segmentations.csv')

    print(masks.shape[0], 'masks found')
    print(masks['ImageId'].value_counts().shape[0], 'unique images found')
    masks = masks.drop(masks[masks.EncodedPixels.isnull()].sample(70000, random_state=42).index)
    print(masks.shape[0], 'after drop')
    unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
    # unique_img_ids.hist(bins=15)
    # plt.show()
    train_ids, valid_ids = train_test_split(unique_img_ids,
                                            test_size=0.05,
                                            stratify=unique_img_ids['counts'],
                                            random_state=42
                                            )
    # print(train_ids.head())
    # print(len(unique_img_ids))
    # print(unique_img_ids['ImageId'].value_counts().shape[0])
    train_ids['ImageId'].to_csv('../../data/split_list/train_ids_b1.csv',header=None,index=None)
    valid_ids['ImageId'].to_csv('../../data/split_list/val_ids_b1.csv', header=None,index=None)

def sample_ships(in_df, base_rep_val=1500):

    #print(in_df.head(20))
    if in_df['grouped_ship_count'].values[0]==0:
        return in_df.sample(base_rep_val//2,random_state=1337) # even more strongly undersample no ships
    # elif in_df['grouped_ship_count'].values[0]==1:
    #     return in_df.sample(base_rep_val*2, random_state=1337)
    else:
        return in_df.sample(base_rep_val, replace=(in_df.shape[0]<base_rep_val),random_state=1337)


def prepare_balance2():
    masks = pd.read_csv('../../data/v2/train_ship_segmentations_v2.csv')
    sp_list = pd.read_csv('../../data/v2/split_list/train.csv',header=None)
    sp_list.columns = ['ImageId']
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['grouped_ship_count'] = unique_img_ids['ships'].map(lambda x:(x+1)//2).clip(0,7)
    unique_img_ids = pd.merge(unique_img_ids,sp_list,on='ImageId',how='inner')
    # print(unique_img_ids.shape[0])
    print(unique_img_ids.shape[0])
    #print(len(unique_img_ids))

    #print(unique_img_ids.head(5))
    balanced_train_df = unique_img_ids.groupby('grouped_ship_count',as_index=False).apply(sample_ships)#.reset_index()
    # balanced_train_df = balanced_train_df.drop('grouped_ship_count')
    print(balanced_train_df.shape[0])
    #print(balanced_train_df.head(20))
    # print(len(balanced_train_df['ImageId']))
    #unique_all_ids_balance =balanced_train_df['ImageId']

    # train_ids, valid_ids = train_test_split(balanced_train_df,
    #                                         test_size=0.2,
    #                                         stratify=balanced_train_df['grouped_ship_count'],
    #                                         random_state=1337)
    #
    # print(len(train_ids))
    # print(len(valid_ids))

    # print(train_ids['ImageId'].value_counts().shape[0])
    #
    # unique_img_ids[unique_img_ids['grouped_ship_count']>=2]['grouped_ship_count'].hist(bins=15)
    # plt.show()

    # balanced_train_df[balanced_train_df['ships']>=0]['ships'].hist(bins=15)
    # plt.show()

    # print(train_ids.head())
    # print(len(unique_img_ids))
    # print(unique_img_ids['ImageId'].value_counts().shape[0])
    balanced_train_df['ImageId'].to_csv('../../data/v2/split_list/train_balance_v2.csv',header=None,index=None)
    # valid_ids['ImageId'].to_csv('../../data/split_list/val_ids_mb.csv', header=None,index=None)


if __name__ == '__main__':
    prepare_balance2()




