# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder



@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        
#         print("------------prefix-------------", prefix, "*****prefix******")  #This will give "sem_seg_head"
#         print(HOHIOH)
        
        version = local_metadata.get("version", None)
        
        # print('-----------------', state_dict.keys(), "******************")
        # print(HOIHOI)
        
        #########################################################################################
        
        
        # state_dict.update({"sem_seg_head.predictor.prompt_feat.1.weight": state_dict["sem_seg_head.predictor.prompt_feat.0.weight"]})
        
        # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.0.bias": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.0.bias"],
        #                    "sem_seg_head.predictor.class_embed.cls.3.layers.0.weight": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.0.weight"],
        #                    "sem_seg_head.predictor.class_embed.cls.3.layers.1.bias": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.1.bias"],
        #                    "sem_seg_head.predictor.class_embed.cls.3.layers.1.weight": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.1.weight"],
        #                    "sem_seg_head.predictor.class_embed.cls.3.layers.2.bias": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.2.bias"],
        #                    "sem_seg_head.predictor.class_embed.cls.3.layers.2.weight": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.2.weight"]})
        
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.0.bias": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.0.bias"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.0.weight": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.0.weight"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.1.bias": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.1.bias"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.1.weight": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.1.weight"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.2.bias": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.2.bias"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.2.weight": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.2.weight"]})
        
        
        # state_dict.update({"sem_seg_head.predictor.prompt_embed.1.0.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.0.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.1.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.1.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.2.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.2.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.3.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.3.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.4.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.4.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.5.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.5.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.6.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.6.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.7.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.7.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.8.weight": state_dict["sem_seg_head.predictor.prompt_embed.0.8.weight"]})
        
        
        # state_dict.update({"sem_seg_head.predictor.prompt_mask_embed.1.layers.0.bias": state_dict["sem_seg_head.predictor.prompt_mask_embed.0.layers.0.bias"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.0.weight": state_dict["sem_seg_head.predictor.prompt_mask_embed.0.layers.0.weight"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.1.bias": state_dict["sem_seg_head.predictor.prompt_mask_embed.0.layers.1.bias"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.1.weight": state_dict["sem_seg_head.predictor.prompt_mask_embed.0.layers.1.weight"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.2.bias": state_dict["sem_seg_head.predictor.prompt_mask_embed.0.layers.2.bias"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.2.weight": state_dict["sem_seg_head.predictor.prompt_mask_embed.0.layers.2.weight"]})





                ############################### STEP 5 ##################################
        # state_dict.update({"sem_seg_head.predictor.prompt_feat.4.weight": state_dict["sem_seg_head.predictor.prompt_feat.3.weight"]})
        
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.0.bias": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.0.bias"],
        # #                    "sem_seg_head.predictor.class_embed.cls.3.layers.0.weight": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.0.weight"],
        # #                    "sem_seg_head.predictor.class_embed.cls.3.layers.1.bias": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.1.bias"],
        # #                    "sem_seg_head.predictor.class_embed.cls.3.layers.1.weight": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.1.weight"],
        # #                    "sem_seg_head.predictor.class_embed.cls.3.layers.2.bias": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.2.bias"],
        # #                    "sem_seg_head.predictor.class_embed.cls.3.layers.2.weight": state_dict["sem_seg_head.predictor.class_embed.cls.2.layers.2.weight"]})
        
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.0.bias": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.0.bias"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.0.weight": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.0.weight"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.1.bias": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.1.bias"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.1.weight": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.1.weight"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.2.bias": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.2.bias"]})
        # # state_dict.update({"sem_seg_head.predictor.class_embed.cls.3.layers.2.weight": state_dict["sem_seg_head.predictor.class_embed.cls.3.layers.2.weight"]})
        
        
        # state_dict.update({"sem_seg_head.predictor.prompt_embed.1.0.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.0.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.1.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.1.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.2.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.2.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.3.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.3.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.4.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.4.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.5.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.5.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.6.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.6.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.7.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.7.weight"],
        #                    "sem_seg_head.predictor.prompt_embed.1.8.weight": state_dict["sem_seg_head.predictor.prompt_embed.1.8.weight"]})
        
        
        # state_dict.update({"sem_seg_head.predictor.prompt_mask_embed.1.layers.0.bias": state_dict["sem_seg_head.predictor.prompt_mask_embed.1.layers.0.bias"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.0.weight": state_dict["sem_seg_head.predictor.prompt_mask_embed.1.layers.0.weight"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.1.bias": state_dict["sem_seg_head.predictor.prompt_mask_embed.1.layers.1.bias"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.1.weight": state_dict["sem_seg_head.predictor.prompt_mask_embed.1.layers.1.weight"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.2.bias": state_dict["sem_seg_head.predictor.prompt_mask_embed.1.layers.2.bias"],
        #                    "sem_seg_head.predictor.prompt_mask_embed.1.layers.2.weight": state_dict["sem_seg_head.predictor.prompt_mask_embed.1.layers.2.weight"]})
        
        # for k in state_dict:
        #     # if k.startswith("sem_seg_head.predictor.class_embed"):
            
        #     if k.startswith("sem_seg_head.predictor.prompt"):
        #         # print(state_dict[k], "--------------state_dict_key-------------------") # This gives the KEY......
        #         print("-------------", k, "*************")
        #         # print("-------------", int(k.split('.')[-4]), "*************")
                
        #         # state_dict["sem_seg_head.predictor.class_embed.cls.0.layers.0.weight"]
                
        #         # print("---------------", k, "***********")
        # print(HIOHOIH)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = transformer_predictor.num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                # print("-------", self.transformer_in_feature, "****")
                # print(OHIH)
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                # print("-------", self.transformer_in_feature, "***FEATURES*")
                # print(JOIJEOIJ)
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions
