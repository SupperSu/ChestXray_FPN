import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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

def plot_heatmap(image, gt,label, model, layer):
    """
    function to plot the heat map on original transformed input image
    Args:
        x: the 1x3x512x512 pytorch tensor file that represents the NIH CXR should be consistent with cfg.CROP_SIZE
        label:user-supplied label you wish to get class activation map for; must be in FINDINGS list
        model: densenet121 trained on NIH CXR data
        layer: which layer's heat map to extract
    Returns:
        cam_torch: 224x224 torch tensor containing activation map
    """
    label_index = -1
    for i in range(14):
        if CLASS_NAMES[i] == label:
            label_index = i
            break
    assert label_index != -1
    res, fea = model(image)
    choosen = fea[layer]
    choosen = torch.sum(choosen, dim=1).unsqueeze(0)
    size = choosen.size()[2:]
    # bilinear pooling to make it becomes original size

    choosen = F.interpolate(choosen, list(size), mode="bilinear")
    raw_cam = choosen.detach()
    # create predictions for label of interest and all labels
    list_res = res[1][0].data.tolist()
    predx = ['%.3f' % elem for elem in list_res]

    fig, (showcxr, heatmap) = plt.subplots(ncols=2, figsize=(14, 5))

    hmap = sns.heatmap(raw_cam.squeeze(),
                       cmap='viridis',
                       alpha=0.3,  # whole heatmap is translucent
                       zorder=2, square=True, vmin=-5, vmax=5
                       )
    cxr = image.squeeze(0).permute(1,2,0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # cxr = cxr * std + mean
    cxr = np.clip(cxr, 0, 1)
    LABEL = label
    # print cxr.size()
    print cxr.shape
    hmap.imshow(cxr,
                aspect=hmap.get_aspect(),
                extent=hmap.get_xlim() + hmap.get_ylim(),
                zorder=1)  # put the map under the heatmap
    hmap.axis('off')
    hmap.set_title("P(" + LABEL + ")=" + str(predx[label_index]))

    showcxr.imshow(cxr)
    showcxr.axis('off')
    showcxr.set_title("Heat Map")
    plt.show()

    preds_concat = pd.concat([pd.Series(CLASS_NAMES), pd.Series(predx), pd.Series(gt.numpy().astype(bool)[0])], axis=1)
    preds = pd.DataFrame(data=preds_concat)
    preds.columns = ["Finding", "Predicted Probability", "Ground Truth"]
    preds.set_index("Finding", inplace=True)
    preds.sort_values(by='Predicted Probability', inplace=True, ascending=False)
    return preds

if __name__ == '__main__':
    # passed
    import torch
    print torch.__version__
    from model import config
    import torch.optim as optim
    from model.model import build_extractor_fpn
    cfg = config.Config
    mdl = build_extractor_fpn(cfg)
    # optimizer = optim.SGD(list(mdl.parameters()), lr=0.0001)
    image = torch.randn(2,3,512,512)
    input = image[0, :, :, :].unsqueeze(0)

    gt = torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    mdl[1].plot_mode()
    plot_heatmap(input, gt,'Hernia', mdl, 1)
