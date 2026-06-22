# VISTA-CLIP (CVPR-W 2025)

**VISTA-CLIP: Visual Incremental Self-Tuned Adaptation for Efficient Continual Panoptic Segmentation** <br />
[Manjunath D](https://manjuphoenix.github.io/)<sup>*1</sup>, [Shrikar Madhu]()<sup>*2</sup>, [Aniruddh Sikdar](https://aniruddhsikdar1.github.io/)<sup>*3</sup>, [Suresh Sundaram](https://scholar.google.com/citations?user=5iAMbhMAAAAJ&hl=en)<sup>1,3</sup><br>

<sup>1</sup> <sub>Dept. of Aerospace Engineering, Indian Institute of Science, Bengaluru, India</sub><br />
<sup>2</sup> <sub>Kotak IISc AI-ML Centre (KIAC), Indian Institute of Science, Bengaluru, India</sub><br />
<sup>3</sup> <sub>Robert Bosch Centre for Cyber Physical Systems, Indian Institute of Science, Bengaluru, India</sub><br />

[![](https://img.shields.io/badge/CVPR_W-2025-blue?logoColor=blue&color=blue)](https://openaccess.thecvf.com/content/CVPR2025W/DG-EBF/papers/D_VISTA-CLIP_Visual_Incremental_Self-Tuned_Adaptation_for_Efficient_Continual_Panoptic_Segmentation_CVPRW_2025_paper.pdf)

</div>


## Introduction
Continual learning seeks to mimic human adaptability by progressively acquiring new knowledge without forgetting previous knowledge. Although deep neural networks per-form well on static tasks, they are prone to catastrophic forgetting, where learning new classes significantly de-grades performance on earlier ones. While most research has focused on classification, detection, and semantic segmentation, comprehensive continual learning frameworks for panoptic segmentation remain largely unexplored. To bridge this gap, we propose VISTA-CLIP (Visual Incremental Self-Tuned Adaptation) that balances plasticity and stability. VISTA-CLIP introduces three key components: (1) Feature Space Exploratory Learnable Perturbations (FLEX), for enhancing feature generalization via learnable noise injection, (2) Visual prompt learning applied at the input level, enabling dynamic adaptation to novel classes without altering core network parameters, and (3) textual feature integration from the CLIP text encoder into class specific prompt embeddings to improve plasticity. Extensive experiments validate the effectiveness of VISTA-CLIP, delivering an average performance gain of 1.2% across all tasks and a notable 2.5% improvement in the challenging 50-50 split (50 base classes followed by 50 incremental). These results position VISTA-CLIP as a scalable and efficient solution for continual panoptic segmentation, pushing the boundaries of lifelong learning in dynamic, real-world settings


## Installation

Our implementation is based on [ECLIPSE](https://github.com/clovaai/ECLIPSE) and [Mask2Former](https://github.com/facebookresearch/Mask2Former).

Please check the [installation instructions](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) and [dataset preparation](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md).

You can see our core implementation from
- `mask2former/maskformer_model.py`
- `mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py`

## Quick Start

1. Step t=0: Training the model for base classes (you can skip this process if you use pre-trained weights.)
2. Step t>1: Training the model for novel classes with ECLIPSE

Run bash script/respective_sh file

<!--
|   Scenario   |  Script   |    Step-0 Weight    |  Final Weight |
| :----------------: | :----------------: | :------:  | :------:  |
| ADE20K-Panoptic 100-5  | `bash script/ade_ps/100_5.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_step0.pth) | [step10](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_5_final.pth)|
| ADE20K-Panoptic 100-10  | `bash script/ade_ps/100_10.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_step0.pth) | [step5](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_10_final.pth)|
| ADE20K-Panoptic 100-50  | `bash script/ade_ps/100_50.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_step0.pth) | [step1](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_100_50_final.pth)|
| | | | |
| ADE20K-Panoptic 50-10  | `bash script/ade_ps/50_10.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_step0.pth) | [step10](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_10_final.pth)|
| ADE20K-Panoptic 50-20  | `bash script/ade_ps/50_20.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_step0.pth) | [step5](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_20_final.pth)|
| ADE20K-Panoptic 50-50  | `bash script/ade_ps/50_50.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_step0.pth) | [step2](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ps_50_50_final.pth)|
| | | | |
| ADE20K-Semantic 100-5  | `bash script/ade_ss/100_5.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_step0.pth) | [step10](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_5_final.pth)|
| ADE20K-Semantic 100-10  | `bash script/ade_ss/100_10.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_step0.pth) | [step5](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_10_final.pth)|
| ADE20K-Semantic 100-50  | `reproduce error` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_step0.pth) | [step1](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/ade_ss_100_50_final.pth)|
| | | | |
| COCO-Panoptic 83-5  | `bash script/coco_ps/83_5.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/coco_ps_83_step0.pth) | [step10](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/coco_ps_83_5_final.pth)|
| COCO-Panoptic 83-10  | `bash script/coco_ps/83_10.sh` |  [step0](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/coco_ps_83_step0.pth) | [step5](https://github.com/clovaai/ECLIPSE/releases/download/ckpt/coco_ps_83_10_final.pth)|
-->


## How to Cite
```
@inproceedings{manjunath2025vista,
  title={VISTA-CLIP: Visual Incremental Self-Tuned Adaptation for Efficient Continual Panoptic Segmentation},
  author={Manjunath, D and Madhu, Shrikar and Sikdar, Aniruddh and Sundaram, Suresh},
  booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={6557--6565},
  year={2025},
  organization={IEEE}
}

```

## License

```
VISTA-CLIP
Copyright (c) 2025-present AIRL, IISc, Bengaluru.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
```
