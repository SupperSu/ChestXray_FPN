import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image
import os
idx2dis = {0: 'Atelectasis',
 1: 'Cardiomegaly',
 2: 'Consolidation',
 3: 'Edema',
 4: 'Effusion',
 5: 'Emphysema',
 6: 'Fibrosis',
 7: 'Hernia',
 8: 'Infiltration',
 9: 'Mass',
 10: 'No Finding',
 11: 'Nodule',
 12: 'Pleural_Thickening',
 13: 'Pneumonia',
 14: 'Pneumothorax'}
class ChestXrayDataSet(Dataset):
    def __init__(self, label_path, pic_list, image_path,transform=None):
        with (open(label_path, 'rb')) as f:
            self.labels = pickle.load(f)
        with open(pic_list, "r") as f:
            self.pic_list = [i.strip() for i in f.readlines()]
        self.root = image_path
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        picName = self.pic_list[index]
        label = self.labels[index]
        # image = torch.randn([256,256]).type(torch.FloatTensor)
        image = Image.open(os.path.join(self.root, picName))
        image = image.convert(mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.from_numpy(label).type(torch.FloatTensor)
        return image, label
    def __len__(self):
        return len(self.pic_list)

def get_data_loader(batch_size, label_path, pic_list, image_path, shuffle = False,transform=None):
    dataset = ChestXrayDataSet(label_path, pic_list, image_path, transform)
    return DataLoader(dataset, batch_size= batch_size, shuffle = shuffle)
if __name__ == '__main__':
    # checked
    label_path = './dataset/train_val_y_onehot.pkl'
    pic_list = './dataset/train_val_list.txt'
    image_path = './dataset/image'
    loader = get_data_loader(12, label_path, pic_list,image_path, True)
    for i ,(image, labels) in enumerate(loader):
        print (labels)