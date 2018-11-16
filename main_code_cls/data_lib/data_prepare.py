import pandas as pd

def make_cls_label():
    masks = pd.read_csv('/home/don/documents/airbus/data/v2/train_ship_segmentations_v2.csv')
    masks['ships'] = masks['EncodedPixels'].map(lambda c_row:1 if isinstance(c_row,str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships':'sum'}).reset_index()

    sp_list = pd.read_csv('/home/don/documents/airbus/data/v2/split_list/inp_val.csv',header=None)
    sp_list.columns = ['ImageId']
    unique_img_ids = pd.merge(unique_img_ids,sp_list)
    unique_img_ids['is_ship'] = unique_img_ids['ships'].map(lambda x:1.0 if x>0 else 0.0)

    print(unique_img_ids.head(20))
    unique_img_ids.to_csv('/home/don/documents/airbus/data/v2/cls_split_list/is_ship_inp_val.csv',index=None)

if __name__ == '__main__':
    make_cls_label()