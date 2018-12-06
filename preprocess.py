import numpy as np
import pandas as pd
from sklearn import preprocessing
def split(train_valid, num_valid = 7000):
    with open(train_valid, "r") as f:
        train_valid_list = [i.strip() for i in f.readlines()]
    valid_list = np.random.choice(train_valid_list, num_valid, replace=False).tolist()
    valid_list = set(valid_list)
    train_valid_list = set(train_valid_list)
    train_list =list(train_valid_list - valid_list)
    valid_list =list(valid_list)
    return train_list, valid_list
def prepare_one_hot_encoder(img_list, label_csv_path):
    with open(img_list, "r") as f:
        tmp_list = [i.strip() for i in f.readlines()]
    y = []
    meta_data = pd.read_csv(label_csv_path)
    for pid in tmp_list:
        labels = meta_data.loc[meta_data["Image Index"] == pid, "Finding Labels"]
        tmp = labels.tolist()[0].split("|")
        y.append(tmp)
    encoder = preprocessing.MultiLabelBinarizer()
    encoder.fit(y)
    return encoder

def one_hot_encode(encoder, img_list, label_csv_path):
    with open(img_list, "r") as f:
        tmp_list = [i.strip() for i in f.readlines()]
    y = []
    meta_data = pd.read_csv(label_csv_path)
    for pid in tmp_list:
        labels = meta_data.loc[meta_data["Image Index"] == pid, "Finding Labels"]
        tmp = labels.tolist()[0].split("|")
        y.append(tmp)
    output = encoder.transform(y)
    return output

if __name__ == '__main__':
    import os
    import pickle as pkl
    print os.getcwd()
    # it's very slowly....
    train_valid_list = 'dataset/train_val_list.txt'
    test_list = 'dataset/test_list.txt'
    data_entry = 'dataset/Data_Entry_2017.csv'
    # use train_valid_list to get one hot encoder
    encoder = prepare_one_hot_encoder(train_valid_list, data_entry)
    num_valid = 7000
    t_list, v_list = split(train_valid_list, num_valid)
    with open('dataset/train_list.txt', 'w') as f:
        for item in t_list:
            f.write("%s\n" % item)
    with open('dataset/valid_list.txt', 'w') as f:
        for item in v_list:
            f.write("%s\n" % item)
    t_y_onehot = one_hot_encode(encoder, 'dataset/train_list.txt', data_entry)
    v_y_onehot = one_hot_encode(encoder, 'dataset/valid_list.txt', data_entry)
    te_y_onehot = one_hot_encode(encoder, 'dataset/test_list.txt', data_entry)
    with open("dataset/train_y_onehot.pkl", "wb") as f:
        pkl.dump(t_y_onehot, f)
    with open("dataset/valid_y_onehot.pkl", "wb") as f:
        pkl.dump(v_y_onehot, f)
    with open("dataset/test_y_onehot.pkl", "wb") as f:
        pkl.dump(te_y_onehot, f)