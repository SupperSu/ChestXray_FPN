import numpy as np
import argparse
import torch
from load_dataset import get_data_loader
from model import config
from model import model
from torchvision import transforms
import time
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
def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    i = 0
    j = 0
    for k in range(15): # totally 14 diseases and 1 label for having disease or not
        if k == 10:
            j +=1
            continue
        AUROCs.append(roc_auc_score(gt_np[:, j], pred_np[:, i]))
        i+=1
        j+=1
    return AUROCs

def test(cfg, args):
    label_path_test = './dataset/test_y_onehot.pkl'
    pic_list_test = './dataset/test_list.txt'
    image_path = './dataset/image'
    mdl_path = args.mdl_path
    transform_test = transforms.Compose(
        [
        transforms.RandomCrop(cfg.CROP_SIZE),
        transforms.ToTensor(),
        ]
    )
    test_loader = get_data_loader(cfg.BATCH_SIZE, label_path_test, pic_list_test, image_path,  False, transform_test)
    mdl = model.build_extractor_fpn(cfg)
    mdl.load_state_dict(torch.load(mdl_path))
    if torch.cuda.is_available():
        print "GPU is available, so the model will be trained on GPU"
        mdl.cuda()
    mdl.eval()
    total_test_step = len(test_loader)
    #TODO: could implmement 10 crops evaluation
    for i, (images, labels) in enumerate(test_loader):
        print 'Step [%d/%d]'.format(i, total_test_step)
        res, _ = mdl(images)
        pre = res[0]
        pred = torch.cat((pred, pre), 0)
        gt = torch.cat((gt, labels), 0)
    AUROCs = tools.compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(14):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdl_path', type=str, default='./checkpoints/',
                        help='path for trained encoder')
    cfg = config.Config()
    args = parser.parse_args()
    test(cfg, args)