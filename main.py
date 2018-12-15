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
import gc
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
    image_path = '/data/chestX_images'
    transform_train = transforms.Compose(
        [
        transforms.RandomCrop(cfg.CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    transform_valid = transforms.Compose(
        [
        transforms.FiveCrop(cfg.CROP_SIZE),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ]
    )
    train_loader = get_data_loader(cfg.BATCH_SIZE, label_path_train, pic_list_train, image_path,  False, transform_train)
    valid_loader = get_data_loader(cfg.VAL_BATCH_SIZE, label_path_valid, pic_list_valid, image_path, False,transform_valid)
    mdl = model.build_extractor_fpn(cfg)
    if torch.cuda.is_available():
        print ("GPU is available, so the model will be trained on GPU")
        mdl.cuda()
    evaluator = build_loss_evaluator(cfg)
    optimizer = optim.SGD(mdl.parameters(), lr= cfg.LR)
    check_point = "./checkpoints/"
    since = time.time
    total_train_step = len(train_loader)
    total_valid_step = len(valid_loader)
    num_epoch = cfg.EPOCH
    output_metric_file = './{}_res.log'.format(cfg.NAME)
    for epoch in range(cfg.EPOCH):
        mdl.train()
        print ("Time:{}, Epoch:{}/{}".format(since, epoch, num_epoch))
        # num = 0
        # after = 0
        for i, (images, labels) in enumerate(train_loader):
            # for obj in gc.get_objects():
            #     try:
            #
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             num += 1
            #     except:
            #         pass
            #
            # print("Before feed input into model the number is{} currenty memory usage of gpu is{}".format(num,
            #       torch.cuda.memory_allocated(device= 0)))
            optimizer.zero_grad()
            # print (images.size())
            results,_ = mdl(images.cuda())

            loss = evaluator(results, labels.cuda())
            loss.backward()
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             after +=1
            #     except:
            #         pass
            # print("after feed input into model the number is{} and the currenty memory usage of gpu is{}".format(after,
            #                                                                                                 torch.cuda.memory_allocated(device= 0)))
            optimizer.step()
            if i % cfg.LOG_STEP == 0:
                print ('Epoch {}/{}, Step {}/{}, Loss: {:5.3}'.format(epoch, cfg.EPOCH, i, total_train_step,
                         loss))
        # # after each epoch do validation

        # torch.save(mdl.state_dict(),check_point + cfg.NAME+ "_epoch" + str(epoch)+".pth")
        # mdl.eval()
        # pred = torch.FloatTensor()
        # gt = torch.FloatTensor()
        # print ("Doing Evaluation for Epoch{}".format(epoch))
        # #TODO: add eraly stopping.
        # for i, (images, labels) in enumerate(valid_loader):
        #     bs, ncrops, c, h, w = images.size()
        #     result, _ = mdl(images.view(-1,c,h,w).cuda())
        #     _, disease_pred = result[0], result[1]
        #     # print (disease_pred)
        #     pre = disease_pred.view(bs, ncrops, -1).mean(1).cpu()
        #     # must be data otherwise will memory leak!!!!!!!
        #     pred = torch.cat((pred, pre.data), 0)
        #     gt = torch.cat((gt, labels.data), 0)
        #     # for obj in gc.get_objects():
        #     #     try:
        #     #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #     #             after += 1
        #     #     except:
        #     #         pass
        #     # print("after feed input into model the number is{} and the currenty memory usage of gpu is{}".format(after,
        #     #                                                                                                      torch.cuda.memory_allocated(
        #     #                                                                                                          device=0)))
        #     if i % 125 == 0:
        #         print ('Step {}/{}'.format(i, total_valid_step))
        # AUROCs = tools.compute_AUCs(gt, pred)
        # AUROC_avg = np.array(AUROCs).mean()
        # f = open(output_metric_file, 'a')
        # f.write('epoch:{}\n'.format(epoch))
        # f.write('The average AUROC is {:5.3}\n'.format(AUROC_avg))
        # for i in range(14):
        #     to_write = 'The AUROC of {} is {}\n'.format(CLASS_NAMES[i], AUROCs[i])
        #     f.write(to_write)
        # f.write('---------------------------------------\n')
        # f.close()

if __name__ == '__main__':
    cfg = config.Config()
    train(cfg)