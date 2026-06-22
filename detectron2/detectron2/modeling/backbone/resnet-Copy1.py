#RESNET FOR NP1 & VP EXPERIMENTS

# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
import random

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    "ResNetBlockBase",
    "BasicBlock",
    "BottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "ResNet",
    "make_stage",
    "build_resnet_backbone",
]

#print(hey)


def Normalization_Perturbation_Plus(feat):
    feat_mean = feat.mean((2, 3), keepdim=True)
    ones_mat = torch.ones_like(feat_mean)
    zeros_mat = torch.zeros_like(feat_mean)
    mean_diff = torch.std(feat_mean, 0, keepdim=True)
    mean_scale = mean_diff / mean_diff.max() * 1.5
    alpha = torch.normal(ones_mat, 0.75 * ones_mat)
    beta = 1 + torch.normal(zeros_mat, 0.75 * ones_mat) * mean_scale
    output = alpha * feat - alpha * feat_mean + beta * feat_mean
    return output



def Orig_Normalization_Perturbation( ):
    # feat: input features of size (B, C, H, W)
    feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
    ones_mat = torch.ones_like(feat_mean)
    alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
    # print("----------", alpha.shape, "*********")
    # print(HEY)
    beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
    # print("-------------", alpha.shape, "*************")
    # print("-------------", beta.shape, "*************")
    # print("--------------", beta.shape, "---beta----")
    output = alpha * feat - alpha * feat_mean + beta * feat_mean
    return output # size: B, C, H, W

def Normalization_Perturbation(feat, alpha, beta):      # Send the learned parameters as an arugment
    # feat: input features of size (B, C, H, W)
    feat_mean = feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
    alpha = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
    # print("----------", alpha.shape, "*********")
    # print(HEY)
    beta = torch.normal(ones_mat, 0.75 * ones_mat) # size: B, C, 1, 1
    # print("-------------", alpha.shape, "*************")
    # print("-------------", beta.shape, "*************")
    # print("--------------", beta.shape, "---beta----")
    output = alpha * feat - alpha * feat_mean + beta * feat_mean
    return output


# TRY NEXT...................
# class HybridAdaptiveNormalization(nn.Module):
#     def __init__(self, feat):
#         super().__init__()
#         ###
#         # self.alpha = torch.normal(ones_mat, 0.75 * ones_mat)
#         # self.alpha = nn.Parameter(alpha)
#         ###
#         self.feat = feat
#         self.feat_mean = self.feat.mean((2, 3), keepdim=True) # size: B, C, 1, 1
#         self.ones_mat = torch.ones_like(self.feat_mean)
#         self.alpha = torch.normal(self.ones_mat, 0.75 * self.ones_mat) # size: B, C, 1, 1
#         self.beta = torch.normal(self.ones_mat, 0.75 * self.ones_mat) # size: B, C, 1, 1
#         self.alpha = nn.Parameter(self.alpha, requires_grad=True)
#         self.beta = nn.Parameter(self.beta, requires_grad=True)
#         # self.alpha = nn.Parameter(torch.ones(2, num_channels, 1, 1))
#         # self.beta = nn.Parameter(torch.ones(2, num_channels, 1, 1))
#         # self.random_std = random_std
        
#     def forward(self, feat):
#         feat_mean = feat.mean((2, 3), keepdim=True)
#         # print("----------", self.alpha.shape, "**************")
#         # print(HEY)
        
#         if self.training:
#             # Calculate adaptive scaling
#             # output = self.alpha * feat - self.alpha * feat_mean + self.beta * feat_mean
#             output = self.alpha * (feat - feat_mean) + self.beta * feat_mean
#             # mean_diff = torch.std(feat_mean, 0, keepdim=True)
#             # mean_scale = mean_diff / mean_diff.max() * 1.5
            
#             # # Apply random perturbations with learned base values
#             # random_alpha = torch.normal(self.alpha, self.random_std * torch.ones_like(self.alpha))
#             # random_beta = self.beta + torch.normal(0, self.random_std * mean_scale)
            
#             # output = random_alpha * (feat - feat_mean) + random_beta * feat_mean
#         else:
#             # Use learned parameters during inference
#             output = self.alpha * (feat - feat_mean) + self.beta * feat_mean
            
#         return output
    
    
class HybridAdaptiveNormalization(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # Initialize with dummy values
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1), requires_grad=False).data.float()
        self.beta = nn.Parameter(torch.ones(1, num_channels, 1, 1), requires_grad=False).data.float()
        self.initialized = False  # Flag to track initialization
        
    def initialize_parameters(self, feat):
        # Initialize parameters based on actual feature tensor
        feat_mean = feat.mean((2, 3), keepdim=True)
        ones_mat = torch.ones_like(feat_mean)
        
        # Create new parameters with proper size
        with torch.no_grad():
            self.alpha.data = torch.normal(ones_mat, 0.75 * ones_mat)
            self.beta.data = torch.normal(ones_mat, 0.75 * ones_mat)
        
        self.initialized = True
    
    def forward(self, feat):
        # Initialize parameters on first forward pass
        if not self.initialized:
            self.initialize_parameters(feat)
            
        
        feat_mean = feat.mean((2, 3), keepdim=True)
        
        if self.training:
            output = self.alpha * (feat - feat_mean) + self.beta * feat_mean
        else:
            output = self.alpha * (feat - feat_mean) + self.beta * feat_mean
            
        return output
    
    
# class HybridAdaptiveNormalization(nn.Module):
#     def __init__(self, num_channels, random_std=0.75,feat):
#         super().__init__()
        
#         self.alpha = nn.Parameter(torch.ones(2, num_channels, 1, 1))
#         self.beta = nn.Parameter(torch.ones(2, num_channels, 1, 1))
#         self.random_std = random_std
        
#     def forward(self, feat):
#         feat_mean = feat.mean((2, 3), keepdim=True)
        
#         if self.training:
#             # Calculate adaptive scaling
#             mean_diff = torch.std(feat_mean, 0, keepdim=True)
#             mean_scale = mean_diff / mean_diff.max() * 1.5
            
#             # Apply random perturbations with learned base values
#             random_alpha = torch.normal(self.alpha, self.random_std * torch.ones_like(self.alpha))
#             random_beta = self.beta + torch.normal(0, self.random_std * mean_scale)
            
#             output = random_alpha * (feat - feat_mean) + random_beta * feat_mean
#         else:
#             # Use learned parameters during inference
#             output = self.alpha * (feat - feat_mean) + self.beta * feat_mean
            
#         return output



class BasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):        
        # print("88888*************", x.shape)
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(CNNBlockBase):
    """
    Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>`
    in the 3x3 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
    ):
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block),
    with a conv, relu and max_pool.
    """

    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


# class ResNet(Backbone):
#     """
#     Implement :paper:`ResNet`.
#     """

#     def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
#         """
#         Args:
#             stem (nn.Module): a stem module
#             stages (list[list[CNNBlockBase]]): several (typically 4) stages,
#                 each contains multiple :class:`CNNBlockBase`.
#             num_classes (None or int): if None, will not perform classification.
#                 Otherwise, will create a linear layer.
#             out_features (list[str]): name of the layers whose outputs should
#                 be returned in forward. Can be anything in "stem", "linear", or "res2" ...
#                 If None, will return the output of the last layer.
#             freeze_at (int): The number of stages at the beginning to freeze.
#                 see :meth:`freeze` for detailed explanation.
#         """
#         super().__init__()
#         self.stem = stem
#         self.num_classes = num_classes
#         self.count = 1

#         current_stride = self.stem.stride
#         self._out_feature_strides = {"stem": current_stride}
#         self._out_feature_channels = {"stem": self.stem.out_channels}

#         self.stage_names, self.stages = [], []

#         if out_features is not None:
#             # Avoid keeping unused layers in this module. They consume extra memory
#             # and may cause allreduce to fail
#             num_stages = max(
#                 [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
#             )
#             stages = stages[:num_stages]
#         for i, blocks in enumerate(stages):
#             assert len(blocks) > 0, len(blocks)
#             for block in blocks:
#                 assert isinstance(block, CNNBlockBase), block

#             name = "res" + str(i + 2)
#             stage = nn.Sequential(*blocks)

#             self.add_module(name, stage)
#             self.stage_names.append(name)
#             self.stages.append(stage)

#             self._out_feature_strides[name] = current_stride = int(
#                 current_stride * np.prod([k.stride for k in blocks])
#             )
#             self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
#         self.stage_names = tuple(self.stage_names)  # Make it static for scripting
        
#         # Hybrid NP
#         self.np_l1 = HybridAdaptiveNormalization(256)
#         self.np_l2 = HybridAdaptiveNormalization(512)
#         # self.alpha_l1 = nn.Parameter(torch.ones(2, 256, 1, 1))
#         # self.beta_l1 = nn.Parameter(torch.ones(2, 256, 1, 1))
        
#         # self.alpha_l2 = nn.Parameter(torch.ones(2, 512, 1, 1))
#         # self.beta_l2 = nn.Parameter(torch.ones(2, 512, 1, 1))
        
#         # self.pertub_s1 = HybridAdaptiveNormalization(num_channels=256)
#         # self.pertub_s2 = HybridAdaptiveNormalization(num_channels=512)

#         if num_classes is not None:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#             self.linear = nn.Linear(curr_channels, num_classes)

#             # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
#             # "The 1000-way fully-connected layer is initialized by
#             # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
#             nn.init.normal_(self.linear.weight, std=0.01)
#             name = "linear"

#         if out_features is None:
#             out_features = [name]
#         self._out_features = out_features
#         assert len(self._out_features)
#         children = [x[0] for x in self.named_children()]
#         for out_feature in self._out_features:
#             assert out_feature in children, "Available children: {}".format(", ".join(children))
#         self.freeze(freeze_at)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

#         Returns:
#             dict[str->Tensor]: names and the corresponding features
#         """
#         assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
#         outputs = {}
        
        
        
        
#         # NP code
#         p = random.random()
        
#         x = self.stem(x)
#         if "stem" in self._out_features:
#             outputs["stem"] = x
            
        
#         # print("----", len(self.stages), "******") # Number of stages is 4
#         """ STAGE 1 """
#         x = self.stages[0](x)   # Stage 1
#         # Normalized Perturbation Plus..........  FOR STAGE 1
#         if self.count==1:
#             print("---------------INITIALIZING NP for Res2--------------------")
#             np_l1 = HybridAdaptiveNormalization(x)
        
#         if  p < 0.5 and self.training:
#             # x = Orig_Normalization_Perturbation(x)
#             x = np_l1(x)
#             # print("---------", p, "---NP PLUS R2-------")
#             # x = Normalization_Perturbation_Plus(x)
#             # x = self.pertub_s1(x)
#             # x = Normalization_Perturbation(x, self.alpha_l1, self.beta_l1)
#             print("---------", x.shape, "---NP PLUS R2-------")
#             # print(HEY)
            
        
#         # print(x.shape, "-8-8-Stage 1-****-")
#         if "res2" in self._out_features and p < 0.5:
#             outputs["res2"] = x
#             # print("----outputs R2----")
#         else:
#             outputs["res2"] = x
            
        
#         """ STAGE 2 """
#         x = self.stages[1](x)   # Stage 2
        
#         if self.count==1:
#             print("---------------INITIALIZING NP for Res3--------------------")
#             np_l2 = HybridAdaptiveNormalization(x)
            
#         # Normalized Perturbation Plus............ FOR STAGE 2
#         if  p < 0.5 and self.training:
#             # alpha_shape, beta_shape = Normalization_Perturbation(x)
#             # x = Orig_Normalization_Perturbation(x)
#             x = np_l2(x)
#             # print("---------", p, "---NP PLUS R3-------")
#             # x = Normalization_Perturbation_Plus(x)
#             # x = self.pertub_s2(x)
#             # x = Normalization_Perturbation(x, self.alpha_l2, self.beta_l2)
#             print("---------", x.shape, "---NP PLUS R3-------")
        
        
#         self.count += 1
#         # print(x.shape, "-8-8-Stage 2-****-")
#         # print(HEY)
#         if "res3" in self._out_features and p < 0.5:
#             outputs["res3"] = x
#             # print(HEY)
#             # print("----outputs R3----")
#         else:
#             outputs["res3"] = x
#             # print(HE*Y)
            
#         """ STAGE 3 """
#         x = self.stages[2](x)   # Stage 3
#         print(x.shape, "-8-8-Stage 3-****-")
#         if "res4" in self._out_features:
#             outputs["res4"] = x
            
            
            
#         """ STAGE 4 """
#         x = self.stages[3](x)   # Stage 4
#         print(x.shape, "-8-8-Stage 4-****-")
#         if "res5" in self._out_features:
#             outputs["res5"] = x
            
        
            
#         # if p < 0.5 and self.training:
#         #     for name, stage in zip(self.stage_names, self.stages):
#         #     # NP
#         #         print("------------------", p, "**********************")
#         #         print("------------------", self.stage_names, self.stages, "**********************")
#         #         print(HEY)
#         #         if name == "res2":
#         #             print("----------- RES2", name)
#         #             x = stage(x)
#         #             x = Normalization_Perturbation(x)
#         #         if name == "res3":
#         #             print("----------- RES3", name)
#         #             x = stage(x)
#         #             x = Normalization_Perturbation(x)  
#         #         x = stage(x)
#         #         if name in self._out_features:
#         #             outputs[name] = x
#         #     # print(HEY)
                
#         # else:
#             # for name, stage in zip(self.stage_names, self.stages):
#             # # NP
#             #     x = stage(x)
#             #     if name in self._out_features:
#             #         outputs[name] = x
            
            
        
#         if self.num_classes is not None:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.linear(x)
#             if "linear" in self._out_features:
#                 outputs["linear"] = x
#         return outputs
    
    
 #####AUTO VP#### TAKEN FROM MIT PAPER
 
# class PadPrompter(nn.Module):
#     def __init__(self):
#         super(PadPrompter, self).__init__()
        
#         pad_size = 30
#         # args.prompt_size
#         #image_size = torch.rand((640,640))
#         image_size = 640
#         # args.image_size

        
        
#         self.base_size = image_size - pad_size*2
#         self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
#         self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
#         self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
#         self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

#     def forward(self, x):
#         base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
#         prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
#         prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
#         prompt = torch.cat(x.size(0) * [prompt])
        
#         # print(x.shape,"-----1----")
        
#         # print(prompt.shape,"--------2-----")
#         # output = x + prompt
#         #print(output.shape, "-----3-----")
#         return x + prompt 
    
    
# class FixedPatchPrompter(nn.Module):
#     def __init__(self):
#         super(FixedPatchPrompter, self).__init__()
        
#         prompt_size = 30
#         # args.prompt_size
#         #image_size = torch.rand((640,640))
#         image_size = 640
#         # args.image_size

        
        
#         self.isize = args.image_size
#         self.psize = args.prompt_size
#         self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

#     def forward(self, x):
#         prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
#         prompt[:, :, :self.psize, :self.psize] = self.patch

#         return x + prompt



class FixedPatchPrompter(nn.Module):
    def __init__(self):
        super(FixedPatchPrompter, self).__init__()
        
        prompt_size = 30
        # args.prompt_size
        #image_size = torch.rand((640,640))
        image_size = 640
        # args.image_size

        
        
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt



# class RandomPatchPrompter(nn.Module):
#     def __init__(self, args):
#         super(RandomPatchPrompter, self).__init__()
#         self.isize = args.image_size
#         self.psize = args.prompt_size
#         self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

#     def forward(self, x):
#         x_ = np.random.choice(self.isize - self.psize)
#         y_ = np.random.choice(self.isize - self.psize)

#         prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
#         prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

#         return x + prompt


    
    
    ##Difference between pad size & prompt size


    
        #print(stop)
    ## Need to add Fixed Patch + Random Patch code

# --------------- ECLIPSE WITH LEARNABLE NP -----------------------
# class ResNet(Backbone):
#     """
#     Implement :paper:`ResNet`.
#     """

#     def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
#         """
#         Args:
#             stem (nn.Module): a stem module
#             stages (list[list[CNNBlockBase]]): several (typically 4) stages,
#                 each contains multiple :class:`CNNBlockBase`.
#             num_classes (None or int): if None, will not perform classification.
#                 Otherwise, will create a linear layer.
#             out_features (list[str]): name of the layers whose outputs should
#                 be returned in forward. Can be anything in "stem", "linear", or "res2" ...
#                 If None, will return the output of the last layer.
#             freeze_at (int): The number of stages at the beginning to freeze.
#                 see :meth:`freeze` for detailed explanation.
#         """
#         super().__init__()
#         self.stem = stem
#         self.num_classes = num_classes
#         # self.count = 1

#         # Add HAN modules for first two stages
#         self.han1 = HybridAdaptiveNormalization(self.stem.out_channels)  # After stem
#         self.han2 = HybridAdaptiveNormalization(stages[0][-1].out_channels)  # After res2
        
        
#         current_stride = self.stem.stride
#         self._out_feature_strides = {"stem": current_stride}
#         self._out_feature_channels = {"stem": self.stem.out_channels}

#         self.stage_names, self.stages = [], []

#         if out_features is not None:
#             # Avoid keeping unused layers in this module. They consume extra memory
#             # and may cause allreduce to fail
#             num_stages = max(
#                 [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
#             )
#             stages = stages[:num_stages]
#         for i, blocks in enumerate(stages):
#             assert len(blocks) > 0, len(blocks)
#             for block in blocks:
#                 assert isinstance(block, CNNBlockBase), block

#             name = "res" + str(i + 2)
#             stage = nn.Sequential(*blocks)

#             self.add_module(name, stage)
#             self.stage_names.append(name)
#             self.stages.append(stage)

#             self._out_feature_strides[name] = current_stride = int(
#                 current_stride * np.prod([k.stride for k in blocks])
#             )
#             self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
#         self.stage_names = tuple(self.stage_names)  # Make it static for scripting
        

#         if num_classes is not None:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#             self.linear = nn.Linear(curr_channels, num_classes)

#             # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
#             # "The 1000-way fully-connected layer is initialized by
#             # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
#             nn.init.normal_(self.linear.weight, std=0.01)
#             name = "linear"

#         if out_features is None:
#             out_features = [name]
#         self._out_features = out_features
#         assert len(self._out_features)
#         children = [x[0] for x in self.named_children()]
#         for out_feature in self._out_features:
#             assert out_feature in children, "Available children: {}".format(", ".join(children))
#         self.freeze(freeze_at)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

#         Returns:
#             dict[str->Tensor]: names and the corresponding features
#         """
#         assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
#         outputs = {}
#         p = random.random()
        
#         #print(x.shape, "FASDFASDFNASDFN")
#         #print(stop)
          
          
        ########### MIT PAD PROMPTER with NP #############################
            
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if self.training:
#             temp= PadPrompter()
#             temp= temp.to(device)
#             temp.train()
#             x=temp(x)

#         # print(temp.shape)
#         # print(stop)
        
#         x = self.stem(x)
            
#         if "stem" in self._out_features:
#             outputs["stem"] = x
            
#         """ STAGE 1 """
#         x = self.stages[0](x)   # Stage 1
        
#         if  p < 0.5 and self.training:
#             x = self.han1(x)  # Apply HAN after First stage
#             # print("---------", x.shape, "---NP PLUS R2-------")
        
        
#         outputs["res2"] = x
            
        
#         """ STAGE 2 """
#         x = self.stages[1](x)   # Stage 2
#         if  p < 0.5 and self.training:
#             x = self.han2(x)  # Apply HAN after Second stage
#             # print("---------", x.shape, "---NP PLUS R2-------")
            
#         if "res3" in self._out_features:
#             outputs["res3"] = x
        
        
#         """ STAGE 3 """
#         x = self.stages[2](x)   # Stage 3
#         if "res4" in self._out_features:
#             outputs["res4"] = x
            
            
#         """ STAGE 4 """
#         x = self.stages[3](x)   # Stage 4
#         if "res5" in self._out_features:
#             outputs["res5"] = x
            
        
#         if self.num_classes is not None:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.linear(x)
#             if "linear" in self._out_features:
#                 outputs["linear"] = x
#         return outputs

#     def output_shape(self):
#         return {
#             name: ShapeSpec(
#                 channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
#             )
#             for name in self._out_features
#         }

#     def freeze(self, freeze_at=0):
#         """
#         Freeze the first several stages of the ResNet. Commonly used in
#         fine-tuning.

#         Layers that produce the same feature map spatial size are defined as one
#         "stage" by :paper:`FPN`.

#         Args:
#             freeze_at (int): number of stages to freeze.
#                 `1` means freezing the stem. `2` means freezing the stem and
#                 one residual stage, etc.

#         Returns:
#             nn.Module: this ResNet itself
#         """
#         if freeze_at >= 1:
#             self.stem.freeze()
#         for idx, stage in enumerate(self.stages, start=2):
#             if freeze_at >= idx:
#                 for block in stage.children():
#                     block.freeze()
#         return self

#     @staticmethod
#     def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
#         """
#         Create a list of blocks of the same type that forms one ResNet stage.

#         Args:
#             block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
#                 stage. A module of this type must not change spatial resolution of inputs unless its
#                 stride != 1.
#             num_blocks (int): number of blocks in this stage
#             in_channels (int): input channels of the entire stage.
#             out_channels (int): output channels of **every block** in the stage.
#             kwargs: other arguments passed to the constructor of
#                 `block_class`. If the argument name is "xx_per_block", the
#                 argument is a list of values to be passed to each block in the
#                 stage. Otherwise, the same argument is passed to every block
#                 in the stage.

#         Returns:
#             list[CNNBlockBase]: a list of block module.

#         Examples:
#         ::
#             stage = ResNet.make_stage(
#                 BottleneckBlock, 3, in_channels=16, out_channels=64,
#                 bottleneck_channels=16, num_groups=1,
#                 stride_per_block=[2, 1, 1],
#                 dilations_per_block=[1, 1, 2]
#             )

#         Usually, layers that produce the same feature map spatial size are defined as one
#         "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
#         all be 1.
#         """
#         blocks = []
#         for i in range(num_blocks):
#             curr_kwargs = {}
#             for k, v in kwargs.items():
#                 if k.endswith("_per_block"):
#                     assert len(v) == num_blocks, (
#                         f"Argument '{k}' of make_stage should have the "
#                         f"same length as num_blocks={num_blocks}."
#                     )
#                     newk = k[: -len("_per_block")]
#                     assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
#                     curr_kwargs[newk] = v[i]
#                 else:
#                     curr_kwargs[k] = v

#             blocks.append(
#                 block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
#             )
#             in_channels = out_channels
#         return blocks

#     @staticmethod
#     def make_default_stages(depth, block_class=None, **kwargs):
#         """
#         Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
#         If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
#         instead for fine-grained customization.

#         Args:
#             depth (int): depth of ResNet
#             block_class (type): the CNN block class. Has to accept
#                 `bottleneck_channels` argument for depth > 50.
#                 By default it is BasicBlock or BottleneckBlock, based on the
#                 depth.
#             kwargs:
#                 other arguments to pass to `make_stage`. Should not contain
#                 stride and channels, as they are predefined for each depth.

#         Returns:
#             list[list[CNNBlockBase]]: modules in all stages; see arguments of
#                 :class:`ResNet.__init__`.
#         """
#         num_blocks_per_stage = {
#             18: [2, 2, 2, 2],
#             34: [3, 4, 6, 3],
#             50: [3, 4, 6, 3],
#             101: [3, 4, 23, 3],
#             152: [3, 8, 36, 3],
#         }[depth]
#         if block_class is None:
#             block_class = BasicBlock if depth < 50 else BottleneckBlock
#         if depth < 50:
#             in_channels = [64, 64, 128, 256]
#             out_channels = [64, 128, 256, 512]
#         else:
#             in_channels = [64, 256, 512, 1024]
#             out_channels = [256, 512, 1024, 2048]
#         ret = []
#         for n, s, i, o in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
#             if depth >= 50:
#                 kwargs["bottleneck_channels"] = o // 4
#             ret.append(
#                 ResNet.make_stage(
#                     block_class=block_class,
#                     num_blocks=n,
#                     stride_per_block=[s] + [1] * (n - 1),
#                     in_channels=i,
#                     out_channels=o,
#                     **kwargs,
#                 )
#             )
#         return ret


# ResNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""



#-------------------- ORGINAL CODE ----------------------------------
class ResNet(Backbone):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}

        ########### MIT PAD PROMPTER withOUT NP #############################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.training:
            #temp= PadPrompter()
            #temp= RandomPatchPrompter()
            temp= FixedPatchPrompter()
            #FixedPatchPrompter
            temp= temp.to(device)
            temp.train()
            x=temp(x)

        # print(temp.shape)
        # print(stop)
        
        #x = self.stem(x)
           
                        


        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[CNNBlockBase]: a list of block module.

        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
            )
            in_channels = out_channels
        return blocks

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.

        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.

        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for n, s, i, o in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    out_channels=o,
                    **kwargs,
                )
            )
        return ret


ResNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""


def make_stage(*args, **kwargs):
    """
    Deprecated alias for backward compatibiltiy.
    """
    return ResNet.make_stage(*args, **kwargs)


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)
