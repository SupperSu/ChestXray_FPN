import numpy as np
import torch.optim as optim
import torch
from load_dataset import get_data_loader
from model import config
from model import model
from torchvision import transforms
import time
from model.loss import build_loss_evaluator
from utils import tools
CLASS_NAMES = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema',
 'Fibrosis',
 'Hernia',
 'Infiltration',
 'Mass',
  'No Finding',
  'Nodule',
  'Pleural_Thickening',
  'Pneumonia',
  'Pneumothorax']

def train(cfg):
    label_path_train = './dataset/train_y_onehot.pkl'
    pic_list_train = './dataset/train_list.txt'
    label_path_valid = './dataset/valid_y_onehot.pkl'
    pic_list_valid = './dataset/valid_list.txt'
    image_path = './dataset/image'
    transform_train = transforms.Compose(
        [
        transforms.RandomCrop(cfg.CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    transform_valid = transforms.Compose(
        [
        transforms.RandomCrop(cfg.CROP_SIZE),
        transforms.ToTensor(),
        ]
    )
    train_loader = get_data_loader(cfg.BATCH_SIZE, label_path_train, pic_list_train, image_path,  False, transform_train)
    valid_loader = get_data_loader(cfg.BATCH_SIZE, label_path_valid, pic_list_valid, image_path, False,transform_valid)
    mdl = model.build_extractor_fpn(cfg)
    if torch.cuda.is_available():
        print "GPU is available, so the model will be trained on GPU"
        mdl.cuda()
    evaluator = build_loss_evaluator(cfg)
    optimizer = optim.SGD(mdl.parameters(), lr= cfg.LR)
    check_point = "./checkpoints"
    since = time.time
    total_train_step = len(train_loader)
    total_valid_step = len(valid_loader)
    num_epoch = cfg.EPOCH
    for epoch in range(cfg.EPOCH):
        mdl.train()
        print "Time:{}, Epoch:{}/{}".format(since, epoch, num_epoch)
        for i, (images, labels) in enumerate(train_loader):
            mdl.zero_grad()
            results,_ = mdl(images)
            loss = evaluator(results, labels)
            loss.backward()
            optimizer.step()
            if i % cfg.LOG_STEP == 0:
                print 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f'.format(epoch, cfg.EPOCH, i, total_train_step,
                         loss.data[0])
        # after each epoch do validation
        mdl.eval()
        torch.save(mdl.state_dict(),check_point )
        pred = torch.FloatTensor()
        gt = torch.FloatTensor()
        print "Doing Evaluation for Epoch[%d]".format(epoch)
        # implement ten crop validation.

        #TODO: add eraly stopping.
        for i, (images, labels) in enumerate(valid_loader):
            res, _ = mdl(images)
            pre = res[0]
            pred = torch.cat((pred, pre), 0)
            gt = torch.cat((gt, labels), 0)
            print 'Step [%d/%d]'.format(i, total_valid_step)
        AUROCs = tools.compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(14):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

if __name__ == '__main__':
    cfg = config.Config()
    train(cfg)