from torchvision import models
import torch.nn.functional as F
from torch import nn as nn
from collections import OrderedDict
import torch
class AttFPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(AttFPNPredictor, self).__init__()
        # assert len(in_channels_list) == len(out_channels)
        # save the blocks name
        self.inner_blocks = []
        self.layer_blocks = []
        self.linear_blocks = []
        in_channels_list = cfg.EXTRACTOR_OUTPUT_CHANNELS
        out_channels = cfg.FPN_OUTPUT_CHANNELS

        for idx, in_channels in enumerate(in_channels_list, 1):
            linear_block = "fpn_linear{}".format(idx)
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            if idx == 4:
                linear_module = nn.Linear(out_channels, 2)
            else:
                linear_module = nn.Linear(out_channels, 14)
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3)
            for module in [inner_block_module, layer_block_module]:
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.add_module(linear_block, linear_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            self.linear_blocks.append(linear_block)
    def interpolate(self, x):
        layer = nn.Upsample(scale_factor=2, mode='nearest')
        return layer(x)

    def layer_attention(self, x, linear_block):
        """
        :param x: input feature maps
        :param linear_block: in the current layer, the linear that is used
        :return: weighted feature maps
        """
        batch_size = x.size()[0]
        weight = getattr(self, linear_block).weight
        # (batch_size, class, num_featuremaps)
        weight = weight.unsqueeze(0).expand(batch_size, weight.size()[0], weight.size()[1])
        weight = torch.transpose(weight, dim0=1, dim1=2)  # (batch_size, num_featuremaps, num_class)
        # use max pool to make (14, 512) to (batch, 512, 1) -- intuition: only consider the sailent part. of each disease
        # since we only focus on patient who have disease, so we only attention on that part
        weight = F.max_pool1d(weight, weight.size()[2])
        weight = weight.unsqueeze(3).expand_as(x)
        weighted = weight * x
        return weighted

    def layer_predict(self, x, linear_block):
        """
        :param x: input feature maps
        :param linear_block: in the current layer, the linear that is used
        :return: current level output
        """
        batch_size = x.size()[0]
        logit = F.max_pool2d(x, x.size()[2:]) #
        logit = getattr(self, linear_block)(logit.view(batch_size, -1))
        # print (logit.size())
        output = F.softmax(logit, dim=1)
        return output

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # FPN top level prediction
        att_feature_maps = []
        results = []
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        last_layer = getattr(self, self.layer_blocks[-1])(last_inner)
        output4 = self.layer_predict(last_layer, self.linear_blocks[-1])
        results.append(output4)
        weighted = self.layer_attention(last_inner, self.linear_blocks[-1])
        att_feature_maps.append(weighted)
        for feature, inner_block, layer_block, linear_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1], self.linear_blocks[:-1][::-1]
        ):
            # for local debug
            inner_top_down = self.interpolate(weighted)
            inner_lateral = getattr(self, inner_block)(feature)
            last_inner = inner_lateral + inner_top_down
            last_layer = getattr(self, layer_block)(last_inner)
            output = self.layer_predict(last_layer, linear_block)
            weighted = self.layer_attention(last_inner, linear_block)
            att_feature_maps.append(weighted)
            results.append(output)
        results = tuple(results)
        has_disease, pres = self.post_process(results)
        return (has_disease, pres), att_feature_maps

    def post_process(self, predictions):
        """
        average each layer prediction.
        prepare results to make its format consistent with labels
        :param results: the output from model
        :return: output results which are same format with labels
        """
        has_disease = predictions[0]
        multi_diseases = predictions[1:]
        pres = torch.stack(list(multi_diseases), dim=0)
        pres = torch.mean(pres, dim=0)
        return has_disease, pres
    def plot_mode(self):
        """
        before to plot the heat map of image, one shall call this method.
        """
        self.batch_size = 1
class Extractor(nn.Module):
    def __init__(self, cfg):
        super(Extractor, self).__init__()
        self.output_channels = []
        if (cfg.FEATURE_EXTRACTOR == 'res50'):
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.cfg = cfg

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.model.children()):
            x = layer(x)
            if idx in self.cfg.EXTRACT_LAYER:
                # print (x.shape)
                features.append(x)
        return features

def build_extractor_fpn(cfg):
    extractor = Extractor(cfg)
    fpn = AttFPNPredictor(cfg)
    model = nn.Sequential(OrderedDict([("extractor", extractor), ("fpn", fpn)]))
    return model
if __name__ == '__main__':
    # checked
    import torch
    import config
    import loss
    import torch.optim as optim
    cfg = config.Config
    evalu = loss.build_loss_evaluator(cfg)
    mdl = build_extractor_fpn(cfg)

    optimizer = optim.SGD(list(mdl.parameters()), lr= 0.0001)
    x = torch.randn(2, 3, 512, 512)
    res, fea = mdl(x)
    print (res)
    # gt = torch.FloatTensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    # mdl.zero_grad()
    # loss = evalu(res, gt)
    # loss.backward()
    # optimizer.step()
    # for fe in fea:
    #     print fe.size()