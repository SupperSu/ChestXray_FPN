import torch
import torch.nn as nn
class LossComputation(object):
    def __init__(self, alpha, beta):
        """
        :param alpha: the penalize parameters for incorrections of have diseases classification
        :param beta: the penalize parameters for incorrections of which diseases multi-label
        classification
        """
        self.alpha = alpha
        self.beta = beta
    def prepare_gt(self, ground_truths):
        """
        :param inputs: labels for image
        :return: one_hot encoded has disease label, and 14 diseases labels.
        """
        batch_size = ground_truths.size()[0]
        has_dise_gt = ground_truths[:, 10].view(batch_size, -1)
        # in the dataset the 11th class label is no_finding
        # we need to make gt as one hot encoding here to match the predictions output format
        # the first label is no diesease, the second is having disease
        no_dise_gt = 1 - has_dise_gt
        has_dise_gt = torch.cat([no_dise_gt, has_dise_gt], dim=1)
        sel1 = ground_truths[:, : 10]
        sel2 = ground_truths[:, 11:]
        disease_gt = torch.cat([sel1, sel2], dim=1)
        return has_dise_gt, disease_gt
    def prepare_pred(self, predictions):
        """
        currently we use average the each layers predictions.
        TODO: may use more advanced method to deal with multi layers predictions
        :param predictions: the predictions for each layer,(tuple type)
        :return: has_disease predictions, and 14 other diseases predictions
        """
        # first top layer tell us whether this patients have disease or not
        has_disease = predictions[0]
        multi_diseases = predictions[1:]
        pres = torch.stack(list(multi_diseases), dim=0)
        pres = torch.mean(pres, dim=0)
        return has_disease, pres
        # average the possibility in different layers

    def __call__(self, predictions, ground_truths):
        has_disease_gt, disease_gt = self.prepare_gt(ground_truths)
        has_disease_pre, disease_pre =predictions[0], predictions[1]
        loss1 = nn.BCELoss()
        loss2 = nn.BCELoss()
        l1 = loss1(has_disease_pre, has_disease_gt)
        l2 = loss2(disease_pre, disease_gt)
        return l1 + l2

def build_loss_evaluator(cfg):
    return LossComputation(cfg.ALPHA, cfg.BETA)

if __name__ == '__main__':
    p1 = torch.FloatTensor([[0, 1, 2], [0, 2, 5], [3, 3, 3]])
    p2 = torch.FloatTensor([[1, 1, 1], [2, 2, 2], [4, 4, 4]])
    pre = (p1, p2)
    gt = torch.FloatTensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]])
    import config
    cfg = config.Config
    evaluator = build_loss_evaluator(cfg)
    print evaluator(pre, gt)
