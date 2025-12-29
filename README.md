# 💦 [IEEE TPAMI 2025]  SLURPP Single-step Latent Underwater Restoration with Pretrained Priors

[![Website](https://img.shields.io/badge/%F0%9F%A4%8D%20Project%20-Website-blue)](https://tianfwang.github.io/slurpp/)
[![Paper](doc/badges/badge-pdf.svg)](https://ieeexplore.ieee.org/document/11127006)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/Tianfwang/SLURPP)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

This repository represents the official implementation of the paper titled "Single-Step Latent Diffusion for Underwater Image Restoration".

Team: [Jiayi Wu](https://jiayi-wu-leo.github.io/), [Tianfu Wang](https://tianfwang.github.io/), [Md Abu Bakr Siddique](https://www.linkedin.com/in/bbkrsddque/), [Md Jahidul Islam](https://jahid.ece.ufl.edu/), [Cornelia Fermuller](https://users.umiacs.umd.edu/~fermulcm/), [Yiannis Aloimonos](https://robotics.umd.edu/clark/faculty/350/Yiannis-Aloimonos), [Christopher A. Metzler](https://www.cs.umd.edu/people/metzler).


![](doc/teaser.png)

![](doc/pipeline.png)
	
## Install envrionment
We use conda for our envrionment mangement, use the ```environment.yml``` file to install 
```
pip uninstall torch torchvision xformers -y
pip install torch torchvision xformers
```

## Inference 
run ```python scripts/download_models.py``` to download pretrained models
then run ```sh scripts/inference/infer_real.sh``` to run inference on the set of example test images


## Training

### Step 0: Training Data
Our method accepts any folder containing terrestrial image, along with the generated metric depth map from [Depth-Pro](https://github.com/apple/ml-depth-pro). \
The image and depth map should be in the same folder. The depth map should have the same file name as the image with the ```depth_pro``` suffix. \
In training scripts set SCRATCH_DATA_DIR location for training data. \
Please inspect ```slurpp/datasets/UR_revised_dataloader.py``` for more details.

### Step 1: Diffusion Fine Tuning
diffusion unet training script is in  ```scripts/training/learn.sh``` \
run ```scripts/training/learn.sh <NAME_OF_YAML_CONFIG_FILE>``` 

### Step 2: Cross-Latent Decoder

first run ```scripts/training/infer_stage2.sh``` to generate pairs of diffusion output/gt data \
then run ```scripts/training/stage2.sh``` to train cross-latent decoder using paried data


## Acknowledgements

This code is modified from the following papers, we thank the authors for their work:

Wang Tianfu, Mingyang Xie, Haoming Cai, Sachin Shah, and Christopher A. Metzler. "Flash-split: 2d reflection removal with flash cues and latent diffusion separation." CVPR 2025.

Ke Bingxin, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. "Repurposing diffusion-based image generators for monocular depth estimation." CVPR 2024.


## Citation

```bibtex
@article{wu2025single,
  title={Single-Step Latent Diffusion for Underwater Image Restoration},
  author={Wu, Jiayi and Wang, Tianfu and Siddique, Md Abu Bakr and Islam, Md Jahidul and Fermuller, Cornelia and Aloimonos, Yiannis and Metzler, Christopher A},
  journal={arXiv preprint arXiv:2507.07878},
  year={2025}
}
```

## License

The code and models of this work are licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE)).

