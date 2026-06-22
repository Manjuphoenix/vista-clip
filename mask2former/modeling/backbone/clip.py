#copied from PODA resnet_clip.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import Union, List, Tuple
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from packaging import version

import os
import warnings
import numpy as np
import urllib
import hashlib
from tqdm import tqdm
from typing import Union, List
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from packaging import version
import clip

import pickle


from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from detectron2.modeling.backbone.resnet import ResNet, BasicStem, BasicBlock, DeformBottleneckBlock, BottleneckBlock


# _tokenizer = _Tokenizer()

# def tokenize(texts: Union[str, List[str]], context_length: int = 256, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
#     """
#     Returns the tokenized representation of given input string(s)

#     Parameters
#     ----------
#     texts : Union[str, List[str]]
#         An input string or a list of input strings to tokenize

#     context_length : int
#         The context length to use; all CLIP models use 77 as the context length

#     truncate: bool
#         Whether to truncate the text in case its encoding is longer than the context length

#     Returns
#     -------
#     A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
#     We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
#     """
#     if isinstance(texts, str):
#         texts = [texts]

#     sot_token = _tokenizer.encoder["<|startoftext|>"]
#     eot_token = _tokenizer.encoder["<|endoftext|>"]
#     all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
#     if version.parse(torch.__version__) < version.parse("1.8.0"):
#         result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
#     else:
#         result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

#     for i, tokens in enumerate(all_tokens):
#         if len(tokens) > context_length:
#             if truncate:
#                 tokens = tokens[:context_length]
#                 tokens[-1] = eot_token
#             else:
#                 raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
#         result[i, :len(tokens)] = torch.tensor(tokens)

#     return result



model_urls = {
    'RN50': 'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt',
}


_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    
    # context_length = 256      
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def load_clip_encoder():
    """
    Load CLIP model weights from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the CLIP checkpoint (.pth file)
    
    Returns:
        model: CLIP encoder model with loaded weights
    """

    import gc

    gc.collect()
    torch.cuda.empty_cache()

    with open("/workspace/localscratch/rbcansi/ECLIpse_text/clip_RN50.pkl", 'rb') as f:
        state_dict = pickle.load(f)


    
    # print(state_dict.keys(), "************")
    # print(JEY)

    
    for k, v in state_dict.items():
        if k.startswith("backbone.visual"):
            state_dict[k] = v.to("cpu")
            del state_dict[k]
            # print(state_dict[k])
        elif k.startswith("backbone.transformer"):
            state_dict[k] = v.to("cpu")
        else:
            state_dict[k] = v.to("cpu")

    # print(state_dict, "************")
    # print(JEY)
    # state_dict = torch.load("/workspace/localscratch/rbcansi/ECLIpse_text/clip_RN50.pkl")

    # print("-----------", state_dict.keys(), "*******************")
    # print(OIHJOERER)
    
    # Extract model dimensions from the state_dict
    embed_dim = state_dict["backbone.text_projection"].shape[1]
    context_length = state_dict["backbone.positional_embedding"].shape[0]
    vocab_size = state_dict["backbone.token_embedding.weight"].shape[0]
    transformer_width = state_dict["backbone.ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("backbone.transformer.resblocks")))
    
    # Extract vision architecture parameters
    vision_width = state_dict["backbone.visual.layer1.0.conv1.weight"].shape[0]
    vision_layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]])
    output_width = round((state_dict["backbone.visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = output_width * 32
    
    # Create a config object with required parameters
    from types import SimpleNamespace
    cfg = SimpleNamespace()
    cfg.MODEL = SimpleNamespace()
    cfg.MODEL.CLIP_encoder = SimpleNamespace()
    cfg.MODEL.CLIP_encoder.EMBED_DIM = embed_dim
    cfg.MODEL.CLIP_encoder.IMAGE_RESOLUTION = image_resolution
    cfg.MODEL.CLIP_encoder.VISION_LAYERS = vision_layers
    cfg.MODEL.CLIP_encoder.VISION_WIDTH = vision_width
    cfg.MODEL.CLIP_encoder.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    
    cfg.MODEL.CLIP_text = SimpleNamespace()
    cfg.MODEL.CLIP_text.CONTEXT_LENGTH = context_length
    cfg.MODEL.CLIP_text.VOCAB_SIZE = vocab_size
    cfg.MODEL.CLIP_text.TRANSFORMER_WIDTH = transformer_width
    
    # Create the model
    model = CLIP_encoder(
        cfg=cfg,
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers
    )

    # device_ids = [0,1]

    # state_dict = nn.DataParallel(state_dict, device_ids=device_ids)
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)


    print("---------------", model, "*****************")
    
    # del state_dict
    # gc.collect()
    # torch.cuda.empty_cache()
    
    # # Configure model for multiple GPUs
    # if torch.cuda.is_available():
    #     num_gpus = torch.cuda.device_count()
        
    #     if use_multiple_gpus and num_gpus > 1:
    #         # Memory-efficient approach for multi-GPU
            
    #         # Check if distributed training is initialized
    #         if torch.distributed.is_available() and torch.distributed.is_initialized():
    #             # Use DDP with careful memory management
    #             try:
    #                 local_rank = torch.distributed.get_rank() % num_gpus
    #                 # Set device explicitly and only load relevant model parts to this GPU
    #                 torch.cuda.set_device(local_rank)
    #                 model = model.to(local_rank)
    #                 torch.cuda.empty_cache()  # Clear any temporary memory
                    
    #                 # Wrap with DistributedDataParallel
    #                 model = torch.nn.parallel.DistributedDataParallel(
    #                     model,
    #                     device_ids=[local_rank],
    #                     output_device=local_rank,
    #                     find_unused_parameters=True  # Helps avoid some DDP errors
    #                 )
    #                 print(f"Model distributed using DistributedDataParallel across {torch.distributed.get_world_size()} processes (memory-efficient mode)")
    #             except Exception as e:
    #                 print(f"Error in DDP setup: {e}. Falling back to single GPU.")
    #                 torch.cuda.empty_cache()
    #                 model = model.to('cuda:0')
    #         else:
    #             # Try model sharding with DataParallel
    #             try:
    #                 # Use a more memory-efficient configuration for DataParallel
    #                 # First, determine available GPU memory
    #                 free_memory = []
    #                 for i in range(num_gpus):
    #                     torch.cuda.set_device(i)
    #                     torch.cuda.empty_cache()
    #                     free_memory.append(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i))
                    
    #                 # If we have very limited memory, use fewer GPUs
    #                 usable_gpus = min(num_gpus, max(1, sum(mem > 2e9 for mem in free_memory)))  # Only use GPUs with >2GB free
                    
    #                 if usable_gpus < num_gpus:
    #                     print(f"Limited GPU memory available. Using {usable_gpus} out of {num_gpus} GPUs")
                    
    #                 if usable_gpus > 1:
    #                     model = model.to('cuda:0')  # Move to first GPU
    #                     model = torch.nn.DataParallel(model, device_ids=list(range(usable_gpus)))
    #                     print(f"Model distributed using DataParallel across {usable_gpus} GPUs (memory-efficient mode)")
    #                 else:
    #                     # Fall back to single GPU
    #                     torch.cuda.set_device(0)  # Use the first GPU
    #                     model = model.to('cuda:0')
    #                     print("Limited memory: using only a single GPU")
    #             except Exception as e:
    #                 print(f"Error in DataParallel setup: {e}. Falling back to single GPU.")
    #                 torch.cuda.empty_cache()
    #                 torch.cuda.set_device(0)
    #                 model = model.to('cuda:0')
    #     else:
    #         # Single GPU mode
    #         torch.cuda.set_device(0)
    #         torch.cuda.empty_cache()
    #         model = model.to('cuda:0')
    #         print("Model loaded on a single GPU")
    # else:
    #     print("No GPU available, model loaded on CPU")
    
    # return model

    # Move model to GPU and utilize multiple GPUs if available and requested
    if torch.cuda.is_available():
        if use_multiple_gpus and torch.cuda.device_count() > 1:
            # Check if we're in a distributed training environment
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # Use DistributedDataParallel for distributed training
                local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
                model = model.to(local_rank)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, 
                    device_ids=[local_rank],
                    output_device=local_rank
                )
                print(f"Model distributed using DistributedDataParallel across {torch.distributed.get_world_size()} processes")
            else:
                # Use DataParallel for single-node multi-GPU training
                model = model.to('cuda')
                model = torch.nn.DataParallel(model)
                print(f"Model distributed using DataParallel across {torch.cuda.device_count()} GPUs")
        else:
            # Use a single GPU
            model = model.to('cuda')
            print("Model loaded on a single GPU")
    else:
        print("No GPU available, model loaded on CPU")

    print("-----------------", model, "*****************")
    print(IOHEOI)
    return model



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,dilation=1):
        super().__init__()
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding= dilation, bias=False,dilation= dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
       
        out += identity
        out = self.relu3(out)
        return out

# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads

#     def forward(self, x):
#         x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
#         print(x.shape)
#         print(self.positional_embedding.shape)
#         print(hey)
#         x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )

#         return x[0]
    
    
    
class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)



# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads

#     def forward(self, x):
#         # print("--First---", x.shape, "------")
#         x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
#         # print("--Second---", x.shape, "-----")
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
#         #print(f"x shape: {x.shape}")
#         #print(f"positional_embedding shape: {self.positional_embedding.shape}")
        
#         print("----", x.shape, self.positional_embedding[:, None, :].to(x.dtype).shape, "-------")
#         x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
#         x, _ = F.multi_head_attention_forward(
#             query=x[:1], key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )
#         return x.squeeze(0)

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            print("---IP---", x.shape, "------")
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        print("---STem---", x.shape, "------")
        x = self.layer1(x)  # det2 resnet50: [256, 200, 304]; CLIP resnet50: [256, 56, 56]
        print("---L1---", x.shape, "------")
        outputs['res2'] = x     # if "res2" in self._out_features else None
        x = self.layer2(x)  # det2 resnet50: [512, 100, 152]; CLIP resnet50: [512, 28, 28]
        print("---L2---", x.shape, "------")
        outputs['res3'] = x     # if "res3" in self._out_features else None
        x = self.layer3(x)  # det2 resnet50: [1024, 50, 76]; CLIP resnet50: [1024, 14, 14]
        print("---L3---", x.shape, "------")
        outputs['res4'] = x     # if "res4" in self._out_features else None
        x = self.layer4(x)  # det2 resnet50: [2048, 25, 38]; CLIP resnet50: [2048, 7, 7]
        print("---L4---", x.shape, "------")
        outputs['res5'] = x     # if "res5" in self._out_features else None
        x = self.attnpool(x)

        return outputs
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

# class CLIP_encoder(nn.Module):
#     def __init__(self,
#                 embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                  #replace_stride_with_dilation : list,
#                  ):
#         super().__init__()

        # vision_heads = vision_width * 32 // 64
        # self.visual = ModifiedResNet(
        #     layers=vision_layers,
        #     output_dim=embed_dim,
        #     heads=vision_heads,
        #     input_resolution=image_resolution,
        #     width=vision_width,
        #     #replace_stride_with_dilation= replace_stride_with_dilation
        # )
        
#         self.initialize_parameters()

#     # def initialize_parameters(self):

#     #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#     #         for name, param in resnet_block.named_parameters():
#     #             if name.endswith("bn3.weight"):
#     #                 nn.init.zeros_(param)

#     def initialize_parameters(self):
#         if isinstance(self.visual, ModifiedResNet):
#             if self.visual.attnpool is not None:
#                 std = self.visual.attnpool.c_proj.in_features ** -0.5
#                 nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
#                 nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

#             for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
#                 for name, param in resnet_block.named_parameters():
#                     if name.endswith("bn3.weight"):
#                         nn.init.zeros_(param)

#     @property
#     def dtype(self):
#         return self.visual.conv1.weight.dtype

#     def forward(self, input):
#         return self.visual(input.type(self.dtype))


# class CLIP_encoder(nn.Module):
#     def __init__(cfg,
#                  self,
#                  embed_dim: int,
#                  # vision
#                  image_resolution: int,
#                  vision_layers: Union[Tuple[int, int, int, int], int],
#                  vision_width: int,
#                 #  vision_patch_size: int,
#                  # text
#                  context_length: int,
#                  vocab_size: int,
#                  transformer_width: int,
#                  transformer_heads: int,
#                  transformer_layers: int
#                  ):
#         super().__init__()
        
#         self.cfg = cfg

#         norm = "FrozenBN"
#         # self.context_length = context_length

#         # if isinstance(vision_layers, (tuple, list)):
#         # vision_heads = vision_width * 32 // 64
        # self.visual = ModifiedResNet(
        #     layers=vision_layers,
        #     output_dim=embed_dim,
        #     heads=vision_heads,
        #     input_resolution=image_resolution,
        #     width=vision_width
        # )
        
        
        # freeze_at           = 0
        # out_features        = ['res2', 'res3', 'res4', 'res5']
        # depth               = 50
        # num_groups          = 1
        # width_per_group     = 64
        # bottleneck_channels = num_groups * width_per_group
        # in_channels         = 64
        # out_channels        = 256
        # stride_in_1x1       = False
        # res5_dilation       = 1
        # deform_on_per_stage = [False, False, False, False]
        # deform_modulated    = False
        # deform_num_groups   = 1
        
        
#         norm = self.cfg.MODEL.RESNETS.NORM
#         freeze_at = self.cfg.MODEL.RESNETS.FREEZE_AT
#         out_channels = self.cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
#         stride_in_1x1 = self.cfg.MODEL.RESNETS.STRIDE_IN_1X1
#         res5_dilation = self.cfg.MODEL.RESNETS.RES5_DILATION
#         deform_on_per_stage = self.cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
#         deform_modulated = self.cfg.MODEL.RESNETS.DEFORM_MODULATED
#         deform_num_groups = self.cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
#         depth = self.cfg.MODEL.RESNETS.DEPTH
#         num_groups = self.cfg.MODEL.RESNETS.NUM_GROUPS
#         width_per_group = self.cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        
#         assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

#         num_blocks_per_stage = {
#             18: [2, 2, 2, 2],
#             34: [3, 4, 6, 3],
#             50: [3, 4, 6, 3],
#             101: [3, 4, 23, 3],
#             152: [3, 8, 36, 3],
#         }[depth]

#         if depth in [18, 34]:
#             assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
#             assert not any(
#                 deform_on_per_stage
#             ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
#             assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
#             assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

#         stages = []

#         for idx, stage_idx in enumerate(range(2, 6)):
#             # res5_dilation is used this way as a convention in R-FCN & Deformable Conv paper
#             dilation = res5_dilation if stage_idx == 5 else 1
#             first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
#             stage_kargs = {
#                 "num_blocks": num_blocks_per_stage[idx],
#                 "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
#                 "in_channels": in_channels,
#                 "out_channels": out_channels,
#                 "norm": norm,
#             }
#             # Use BasicBlock for R18 and R34.
#             if depth in [18, 34]:
#                 stage_kargs["block_class"] = BasicBlock
#             else:
#                 stage_kargs["bottleneck_channels"] = bottleneck_channels
#                 stage_kargs["stride_in_1x1"] = stride_in_1x1
#                 stage_kargs["dilation"] = dilation
#                 stage_kargs["num_groups"] = num_groups
#                 if deform_on_per_stage[idx]:
#                     stage_kargs["block_class"] = DeformBottleneckBlock
#                     stage_kargs["deform_modulated"] = deform_modulated
#                     stage_kargs["deform_num_groups"] = deform_num_groups
#                 else:
#                     stage_kargs["block_class"] = BottleneckBlock
                    
#             blocks = ResNet.make_stage(**stage_kargs)
#             in_channels = out_channels
#             out_channels *= 2
#             bottleneck_channels *= 2
#             stages.append(blocks)
#         stem = BasicStem(3, 64, "FrozenBN")
#         # stages = [[BottleneckBlock((shortcut): Conv2d(
#         #         64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#         #     )
#         #     (conv1): Conv2d(
#         #         64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#         #     )
#         #     (conv2): Conv2d(
#         #         64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#         #     )
#         #     (conv3): Conv2d(
#         #         64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#         #     )
#         #     ), BottleneckBlock(
#         #     (conv1): Conv2d(
#         #         256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#         #     )
#         #     (conv2): Conv2d(
#         #         64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#         #     )
#         #     (conv3): Conv2d(
#         #         64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#         #     )
#         #     ), BottleneckBlock(
#         #     (conv1): Conv2d(
#         #         256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#         #     )
#         #     (conv2): Conv2d(
#         #         64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=64, eps=1e-05)
#         #     )
#         #     (conv3): Conv2d(
#         #         64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
#         #         (norm): FrozenBatchNorm2d(num_features=256, eps=1e-05)
#         #     ))]]
#         # self.visual = ResNet(stem, stages)
#         self.visual = ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)
#         # else:
#         # vision_heads = vision_width // 64
#         # self.visual = VisionTransformer(
#         #     input_resolution=image_resolution,
#         #     patch_size=vision_patch_size,
#         #     width=vision_width,
#         #     layers=vision_layers,
#         #     heads=vision_heads,
#         #     output_dim=embed_dim
#         # )

#         self.transformer = Transformer(
#             width=transformer_width,
#             layers=transformer_layers,
#             heads=transformer_heads,
#             attn_mask=self.build_attention_mask()
#         )

#         self.vocab_size = vocab_size
#         self.token_embedding = nn.Embedding(vocab_size, transformer_width)
#         self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
#         self.ln_final = LayerNorm(transformer_width)

#         self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         self.initialize_parameters()







class CLIP_encoder(nn.Module):
    def __init__(self,
                 cfg,  # Add cfg as an argument
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                #  vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        
        self.cfg = cfg # Store cfg as an attribute
        
        norm = "FrozenBN"
        
        # context_length = cfg.MODEL.CLIP_text.CONTEXT_LENGTH

        freeze_at           = 0
        out_features        = ['res2', 'res3', 'res4', 'res5']
        depth               = 50
        num_groups          = 1
        width_per_group     = 64
        bottleneck_channels = num_groups * width_per_group
        in_channels         = 64
        out_channels        = 256
        stride_in_1x1       = False
        res5_dilation       = 1
        deform_on_per_stage = [False, False, False, False]
        deform_modulated    = False
        deform_num_groups   = 1


        out_features        = ['res2', 'res3', 'res4', 'res5'] #This can be a constant


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
                "in_channels": 64 if idx == 0 else out_channels // 2,  # Use in_channels correctly
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
            #in_channels = out_channels  #Correct
            in_channels = out_channels
            out_channels *= 2
            bottleneck_channels *= 2
            stages.append(blocks)

        stem = BasicStem(3, 64, "FrozenBN")


    ###################### CLIP IMG Encoder starts here #########################
        # vision_heads = vision_width * 32 // 64
        
        # self.visual = ModifiedResNet(
        #     layers=vision_layers,
        #     output_dim=embed_dim,
        #     heads=vision_heads,
        #     input_resolution=image_resolution,
        #     width=vision_width,
        #     #replace_stride_with_dilation= replace_stride_with_dilation
        # )
        
        

    ####################### RESNET ENCODER ###############################

        # self.clip_model = load_clip_encoder()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_text_enc = clip.load('RN50', device)[0].encode_text

        # print("---------------", self.clip_model, "****************")
        # print(HOIHIO)

        # self.text_encoder = self.clip_model.encode_text

        
        self.visual = ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        
        # self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        
        
        
        
        
        

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        # mask = torch.empty(self.context_length, self.context_length)
        mask = torch.empty(77, 77)
        # mask = torch.empty(256, 256)      # For text embedding of size (256, 256)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image)
        # return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # text = text.squeeze(1)
        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # print(x.shape, x, "*******************")
        # print(OIHOHI)
        
        # x = torch.squeeze(x)  # [batch_size, n_ctx, d_model]
        
        # x = x + self.positional_embedding.type(self.dtype)
        print("----------------", self.positional_embedding, "*********************")
        print(OIHOIU)
        
        x = x + self.positional_embedding
        # print("------------", x.shape, "***************")
        # print(HIHO)
        
        x = x.permute(1, 0, 2)  # NLD -> LND

        print("----------------_", self.transformer, "***************")
        print(HOIEHOI)
        
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x



    def encode_text_tokens(self, tokenized_text):
        """
        Encode a list of text strings using the CLIP text encoder
        
        Args:
            text_list: List of strings to encode
        
        Returns:
            text_features: Tensor of encoded text features
        """
        # text = text.squeeze(1)
        # text_tokens = tokenize(text_list).to(device="cuda")
        # with torch.no_grad():
        #     text_features = self.clip_model.encode_text(text_tokens)
        # return text_features
        text_feat = self.clip_text_enc(tokenized_text)
        return text_feat




        
    def forward(self, image, text):
        """ input: 
                image: batch of images of shape [B, C, H, W]
                text: list of class names
                
            output:
                image_features: shape [N, C, H, W]
                text_batch: list  of text embeddings for each image"""
        text_batch = []
        
            # print("---------text feat----------", text_features_ind.shape, "*******")
            
        # text_batch = torch.cat(text_batch, dim=1)
        # print("-------------------", text_batch.shape, "*******")
        # print(ljoihoi)
            
        if self.training:
            # with torch.no_grad():
            for i in text:
                # text_token = tokenize(i).to(device="cuda")
                
                # text_features_ind = self.clip_text_enc(text_token)
                # print("----------------", text_token.shape, "*********text token ***************")
                # text_features_ind = self.encode_text_tokens(text_token)
                text_features_ind = self.encode_text(i)
                # text_features_ind = self.clip_text_enc(text_token)
                # print("----------------", text_features_ind.shape, "*********text features ***************")
                text_batch.append(text_features_ind)
                print("--------------", self.encode_text, "****************")
                # print(HOIEHO)
                
                # text_features_ind = self.encode_text(text_token)
                # text_batch.append(text_features_ind)
            # image_features = ResNet.forward(x=image)
            # print("************", len(text_batch), text_batch[0].shape, "-----------------")
            # print(OIOFEF)
            
            ############## RESNET WORKING ##################
            image_features = self.encode_image(image)

            # image_features = self.visual(image)
            # print("--------------------", image_features["res2"].shape, "&*******************")            
            # print("--------------------", image_features["res3"].shape, "&*******************")            
            # print("--------------------", image_features["res4"].shape, "&*******************")            
            # print("--------------------", image_features["res5"].shape, "&*******************")
            # print("************", len(text_batch), text_batch[0].shape, "-----------------")
            # print(OHOIH)
            
            # text = tokenize(text).to(device="cuda")
            # text_features = self.encode_text(text)
            # print("--------img-----------", image_features["res4"].shape, "*************")
            # print("---------text----------", text_features.shape, "*************")
            # print(IHOI)
            
            
            #################### OLD CODE #############################
            # normalized features
            # image_features = image_features / image_features.norm(dim=1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            # logit_scale = self.logit_scale.exp()
            # logits_per_image = logit_scale * image_features @ text_features.t()
            # logits_per_text = logits_per_image.t()

            # shape = [global_batch_size, global_batch_size]
            # return logits_per_image, logits_per_text
            # return image_features, text_features
            return image_features, text_batch
        else:
            # image_features = self.encode_image(image)
            # return image_features
            
            # When inference also considers text information....
            image_features = self.encode_image(image)
            # text = tokenize(text).to(device="cuda")
            # text_features = self.encode_text(text)
            # print("-------------------", image_features["res4"].shape, "*************")
            # print("-------------------", text_features.shape, "*************")
            # print(IHOI)
            # return image_features, text_features
            return image_features, None



 #################### OLD CODE #############################
# @BACKBONE_REGISTRY.register()
# class CLIP_back(CLIP_encoder, Backbone):
#     def __init__(self, cfg, input_shape):
#         embed_dim = cfg.MODEL.CLIP_encoder.EMBED_DIM
#         image_resolution = cfg.MODEL.CLIP_encoder.IMAGE_RESOLUTION
#         vision_layers = cfg.MODEL.CLIP_encoder.VISION_LAYERS
#         vision_width = cfg.MODEL.CLIP_encoder.VISION_WIDTH
#         #replace_stride_with_dilation = cfg.MODEL.CLIP_encoder.REPLACE_STRIDE_WITH_DILATION

#         super().__init__(embed_dim, image_resolution, vision_layers, vision_width)

#         self._out_features = cfg.MODEL.CLIP_encoder.OUT_FEATURES

#         self._out_feature_strides = {
#             "stem": 4,
#             "res2": 4,
#             "res3": 8,
#             "res4": 16,
#             "res5": 32,
#         }
#         self._out_feature_channels = {
#             "res2": 256,
#             "res3": 512,
#             "res4": 1024,
#             "res5": 2048,
#         }

#     def forward(self, input):
#         # print(input.shape)
#         # print(hey)
#         outputs = {}
#         y = super().forward(input)
#         for k in y.keys():
#             if k in self._out_features:
#                 outputs[k] = y[k]
#         return outputs
    
#     def output_shape(self):
#         return {
#             name: ShapeSpec(
#                 channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
#             )
#             for name in self._out_features
#         }


@BACKBONE_REGISTRY.register()
class CLIP_back(CLIP_encoder, Backbone):
    def __init__(self, cfg, input_shape):
        embed_dim = cfg.MODEL.CLIP_encoder.EMBED_DIM
        image_resolution = cfg.MODEL.CLIP_encoder.IMAGE_RESOLUTION
        vision_layers = cfg.MODEL.CLIP_encoder.VISION_LAYERS
        vision_width = cfg.MODEL.CLIP_encoder.VISION_WIDTH
        #replace_stride_with_dilation = cfg.MODEL.CLIP_encoder.REPLACE_STRIDE_WITH_DILATION

        # TESTING PURPOSE
        context_length = cfg.MODEL.CLIP_text.CONTEXT_LENGTH
        vocab_size = cfg.MODEL.CLIP_text.VOCAB_SIZE
        transformer_width = cfg.MODEL.CLIP_text.TRANSFORMER_WIDTH
        transformer_heads = transformer_width // 64
        # transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
        transformer_layers = 12
        
        output_features = ["res2", "res3", "res4", "res5"]
        freeze_at = 0
        
        # super().__init__(embed_dim, image_resolution, vision_layers, vision_width, output_features)

        # TESTING PURPOSE
        # super().__init__(embed_dim, image_resolution, vision_layers, vision_width,
        #                  context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, output_features, freeze_at)
        
        super().__init__(cfg, embed_dim, image_resolution, vision_layers, vision_width,
                         context_length, vocab_size, transformer_width, transformer_heads,
                         transformer_layers)
        
        
        self._out_features = cfg.MODEL.CLIP_encoder.OUT_FEATURES

        self._out_feature_strides = {
            "stem": 4,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": 256,
            "res3": 512,
            "res4": 1024,
            "res5": 2048,
        }

    # TESTING PURPOSE
    def forward(self, input_img, txt_prompt=None):
        if self.training:
            img_logit, text_logits = super().forward(input_img, txt_prompt)
        else:
            img_logit = super().forward(input_img, txt_prompt)
        # TESTING PURPOSE
        if txt_prompt != None:
            img_feat, text_feat = super().forward(input_img, txt_prompt)
            return img_feat, text_feat
        # return img_feat
            
        # TESTING PURPOSE
        else:
            img_feat = super().forward(input_img, txt_prompt)
            # img_feat = super().forward(input_img)
            return img_feat
        for k in img_logit.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs
        
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }



def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


to_remove = [ "positional_embedding","text_projection", "logit_scale", "input_resolution", "context_length", "vocab_size", "transformer.resblocks.0.attn.in_proj_weight", "transformer.resblocks.0.attn.in_proj_bias", "transformer.resblocks.0.attn.out_proj.weight", "transformer.resblocks.0.attn.out_proj.bias", "transformer.resblocks.0.ln_1.weight", "transformer.resblocks.0.ln_1.bias", "transformer.resblocks.0.mlp.c_fc.weight", "transformer.resblocks.0.mlp.c_fc.bias", "transformer.resblocks.0.mlp.c_proj.weight", "transformer.resblocks.0.mlp.c_proj.bias", "transformer.resblocks.0.ln_2.weight", "transformer.resblocks.0.ln_2.bias", "transformer.resblocks.1.attn.in_proj_weight", "transformer.resblocks.1.attn.in_proj_bias", "transformer.resblocks.1.attn.out_proj.weight", "transformer.resblocks.1.attn.out_proj.bias", "transformer.resblocks.1.ln_1.weight", "transformer.resblocks.1.ln_1.bias", "transformer.resblocks.1.mlp.c_fc.weight", "transformer.resblocks.1.mlp.c_fc.bias", "transformer.resblocks.1.mlp.c_proj.weight", "transformer.resblocks.1.mlp.c_proj.bias", "transformer.resblocks.1.ln_2.weight", "transformer.resblocks.1.ln_2.bias", "transformer.resblocks.2.attn.in_proj_weight", "transformer.resblocks.2.attn.in_proj_bias", "transformer.resblocks.2.attn.out_proj.weight", "transformer.resblocks.2.attn.out_proj.bias", "transformer.resblocks.2.ln_1.weight", "transformer.resblocks.2.ln_1.bias", "transformer.resblocks.2.mlp.c_fc.weight", "transformer.resblocks.2.mlp.c_fc.bias", "transformer.resblocks.2.mlp.c_proj.weight", "transformer.resblocks.2.mlp.c_proj.bias", "transformer.resblocks.2.ln_2.weight", "transformer.resblocks.2.ln_2.bias", "transformer.resblocks.3.attn.in_proj_weight", "transformer.resblocks.3.attn.in_proj_bias", "transformer.resblocks.3.attn.out_proj.weight", "transformer.resblocks.3.attn.out_proj.bias", "transformer.resblocks.3.ln_1.weight", "transformer.resblocks.3.ln_1.bias", "transformer.resblocks.3.mlp.c_fc.weight", "transformer.resblocks.3.mlp.c_fc.bias", "transformer.resblocks.3.mlp.c_proj.weight", "transformer.resblocks.3.mlp.c_proj.bias", "transformer.resblocks.3.ln_2.weight", "transformer.resblocks.3.ln_2.bias", "transformer.resblocks.4.attn.in_proj_weight", "transformer.resblocks.4.attn.in_proj_bias", "transformer.resblocks.4.attn.out_proj.weight", "transformer.resblocks.4.attn.out_proj.bias", "transformer.resblocks.4.ln_1.weight", "transformer.resblocks.4.ln_1.bias", "transformer.resblocks.4.mlp.c_fc.weight", "transformer.resblocks.4.mlp.c_fc.bias", "transformer.resblocks.4.mlp.c_proj.weight", "transformer.resblocks.4.mlp.c_proj.bias", "transformer.resblocks.4.ln_2.weight", "transformer.resblocks.4.ln_2.bias", "transformer.resblocks.5.attn.in_proj_weight", "transformer.resblocks.5.attn.in_proj_bias", "transformer.resblocks.5.attn.out_proj.weight", "transformer.resblocks.5.attn.out_proj.bias", "transformer.resblocks.5.ln_1.weight", "transformer.resblocks.5.ln_1.bias", "transformer.resblocks.5.mlp.c_fc.weight", "transformer.resblocks.5.mlp.c_fc.bias", "transformer.resblocks.5.mlp.c_proj.weight", "transformer.resblocks.5.mlp.c_proj.bias", "transformer.resblocks.5.ln_2.weight", "transformer.resblocks.5.ln_2.bias", "transformer.resblocks.6.attn.in_proj_weight", "transformer.resblocks.6.attn.in_proj_bias", "transformer.resblocks.6.attn.out_proj.weight", "transformer.resblocks.6.attn.out_proj.bias", "transformer.resblocks.6.ln_1.weight", "transformer.resblocks.6.ln_1.bias", "transformer.resblocks.6.mlp.c_fc.weight", "transformer.resblocks.6.mlp.c_fc.bias", "transformer.resblocks.6.mlp.c_proj.weight", "transformer.resblocks.6.mlp.c_proj.bias", "transformer.resblocks.6.ln_2.weight", "transformer.resblocks.6.ln_2.bias", "transformer.resblocks.7.attn.in_proj_weight", "transformer.resblocks.7.attn.in_proj_bias", "transformer.resblocks.7.attn.out_proj.weight", "transformer.resblocks.7.attn.out_proj.bias", "transformer.resblocks.7.ln_1.weight", "transformer.resblocks.7.ln_1.bias", "transformer.resblocks.7.mlp.c_fc.weight", "transformer.resblocks.7.mlp.c_fc.bias", "transformer.resblocks.7.mlp.c_proj.weight", "transformer.resblocks.7.mlp.c_proj.bias", "transformer.resblocks.7.ln_2.weight", "transformer.resblocks.7.ln_2.bias", "transformer.resblocks.8.attn.in_proj_weight", "transformer.resblocks.8.attn.in_proj_bias", "transformer.resblocks.8.attn.out_proj.weight", "transformer.resblocks.8.attn.out_proj.bias", "transformer.resblocks.8.ln_1.weight", "transformer.resblocks.8.ln_1.bias", "transformer.resblocks.8.mlp.c_fc.weight", "transformer.resblocks.8.mlp.c_fc.bias", "transformer.resblocks.8.mlp.c_proj.weight", "transformer.resblocks.8.mlp.c_proj.bias", "transformer.resblocks.8.ln_2.weight", "transformer.resblocks.8.ln_2.bias", "transformer.resblocks.9.attn.in_proj_weight", "transformer.resblocks.9.attn.in_proj_bias", "transformer.resblocks.9.attn.out_proj.weight", "transformer.resblocks.9.attn.out_proj.bias", "transformer.resblocks.9.ln_1.weight", "transformer.resblocks.9.ln_1.bias", "transformer.resblocks.9.mlp.c_fc.weight", "transformer.resblocks.9.mlp.c_fc.bias", "transformer.resblocks.9.mlp.c_proj.weight", "transformer.resblocks.9.mlp.c_proj.bias", "transformer.resblocks.9.ln_2.weight", "transformer.resblocks.9.ln_2.bias", "transformer.resblocks.10.attn.in_proj_weight", "transformer.resblocks.10.attn.in_proj_bias", "transformer.resblocks.10.attn.out_proj.weight", "transformer.resblocks.10.attn.out_proj.bias", "transformer.resblocks.10.ln_1.weight", "transformer.resblocks.10.ln_1.bias", "transformer.resblocks.10.mlp.c_fc.weight", "transformer.resblocks.10.mlp.c_fc.bias", "transformer.resblocks.10.mlp.c_proj.weight", "transformer.resblocks.10.mlp.c_proj.bias", "transformer.resblocks.10.ln_2.weight", "transformer.resblocks.10.ln_2.bias", "transformer.resblocks.11.attn.in_proj_weight", "transformer.resblocks.11.attn.in_proj_bias", "transformer.resblocks.11.attn.out_proj.weight", "transformer.resblocks.11.attn.out_proj.bias", "transformer.resblocks.11.ln_1.weight", "transformer.resblocks.11.ln_1.bias", "transformer.resblocks.11.mlp.c_fc.weight", "transformer.resblocks.11.mlp.c_fc.bias", "transformer.resblocks.11.mlp.c_proj.weight", "transformer.resblocks.11.mlp.c_proj.bias", "transformer.resblocks.11.ln_2.weight", "transformer.resblocks.11.ln_2.bias", "token_embedding.weight", "ln_final.weight", "ln_final.bias"]
#


def build_model(state_dict: dict):

    state_dict = torch.load("./clip_RN50.pth")
    print(state_dict.keys(), "______---------------------_________-----")
    print(OHOIH)
    
    embed_dim = state_dict["text_projection"].shape[1]
    for k in to_remove:
        state_dict.pop(k, None)

    counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    image_resolution = output_width * 32

    model = CLIP_encoder(embed_dim, image_resolution, vision_layers, vision_width)

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target