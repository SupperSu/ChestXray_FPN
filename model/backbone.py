from torchvision import models
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

# class FPN(nn.Module):
#     """
#     Module that adds FPN on top of a list of feature maps.
#     The feature maps are currently supposed to be in increasing depth
#     order, and must be consecutive
#     """
#
#     def __init__(self, in_channels_list, out_channels, top_blocks=None):
#         """
#         Arguments:
#             in_channels_list (list[int]): number of channels for each feature map that
#                 will be fed
#             out_channels (int): number of channels of the FPN representation
#             top_blocks (nn.Module or None): if provided, an extra operation will
#                 be performed on the output of the last (smallest resolution)
#                 FPN output, and the result will extend the result list
#         """
#         super(FPN, self).__init__()
#         # assert len(in_channels_list) == len(out_channels)
#         self.inner_blocks = []
#         self.layer_blocks = []
#         for idx, in_channels in enumerate(in_channels_list, 1):
#             inner_block = "fpn_inner{}".format(idx)
#             layer_block = "fpn_layer{}".format(idx)
#             inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
#             layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
#             for module in [inner_block_module, layer_block_module]:
#                 # Caffe2 implementation uses XavierFill, which in fact
#                 # corresponds to kaiming_uniform_ in PyTorch
#                 nn.init.kaiming_uniform_(module.weight, a=1)
#                 nn.init.constant_(module.bias, 0)
#             self.add_module(inner_block, inner_block_module)
#             self.add_module(layer_block, layer_block_module)
#             self.inner_blocks.append(inner_block)
#             self.layer_blocks.append(layer_block)
#         self.top_blocks = top_blocks
#     def interpolate(self, x):
#         layer = nn.Upsample(scale_factor=2, mode='nearest')
#         return layer(x)
#     def forward(self, x):
#         """
#         Arguments:
#             x (list[Tensor]): feature maps for each feature level.
#         Returns:
#             results (tuple[Tensor]): feature maps after FPN layers.
#                 They are ordered from highest resolution first.
#         """
#         last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
#         results = []
#         results.append(getattr(self, self.layer_blocks[-1])(last_inner))
#         for feature, inner_block, layer_block in zip(
#             x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
#         ):
#             # for local debug
#             inner_top_down = self.interpolate(last_inner)
#             # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
#             inner_lateral = getattr(self, inner_block)(feature)
#             # TODO use size instead of scale to make it robust to different sizes
#             # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
#             # mode='bilinear', align_corners=False)
#             last_inner = inner_lateral + inner_top_down
#             results.insert(0, getattr(self, layer_block)(last_inner))
#         if self.top_blocks is not None:
#             last_results = self.top_blocks(results[-1])
#             results.extend(last_results)
#         return tuple(results)
#
#
# class LastLevelMaxPool(nn.Module):
#     def forward(self, x):
#         return [F.max_pool2d(x, 1, 2, 0)]


class ResNet_50(nn.Module):
    def __init__(self, cfg):
        super(ResNet_50, self).__init__()
        self.output_channels = []
        if (cfg.FEATURE_EXTRACTOR == 'res50'):
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.cfg = cfg

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.model.children()):
            print (idx)
            x = layer(x)
            if idx in self.cfg.EXTRACT_LAYER:
                features.append(x)
        return features
#TODO: add vgg 19 backbone

