# Rethinking Mixture of Rain Removal via Depth-Guided Adversarial Learning (NN'2025)

Authors: Yongzhen Wang, Xuefeng Yan, Yanbiao Niu, Lina Gong, Yanwen Guo, and Mingqiang Wei

[[Paper Link]](https://www.sciencedirect.com/science/article/abs/pii/S0893608025006197)

### Abstract

Rainy weather significantly deteriorates the visibility of scene objects, particularly when images are captured through outdoor camera lenses or windshields. Through careful observation of numerous rainy photos, we have discerned that the images are typically affected by various rainwater artifacts such as raindrops, rain streaks, and rainy haze, which impair the image quality from near to far distances, resulting in a complex and intertwined process of image degradation. However, current deraining techniques are limited in their ability to address only one or two types of rainwater, which poses a challenge in removing the mixture of rain (MOR). In this study, we naturally associate scene depth with the MOR effect and propose an effective image deraining paradigm for the Mixture of Rain Removal, termed DEMore-Net. Going beyond the existing deraining wisdom, DEMore-Net is a joint learning paradigm that integrates depth estimation and MOR removal tasks to achieve superior rain removal. The depth information can offer additional meaningful guidance information based on distance, thus better helping DEMore-Net remove different types of rainwater. Moreover, this study explores normalization approaches in image deraining tasks and introduces a new Hybrid Normalization Block (HNB) to enhance the deraining performance of DEMore-Net. Extensive experiments conducted on synthetic datasets and real-world MOR photos fully validate the superiority of DEMore-Net. Code is available at https://github.com/yz-wang/UCL-Dehaze.

#### If you find the resource useful, please cite the following :- )

```
@article{WANG2025107739,
title = {Rethinking mixture of rain removal via depth-guided adversarial learning},
journal = {Neural Networks},
volume = {191},
pages = {107739},
year = {2025},
issn = {0893-6080},
author = {Yongzhen Wang and Xuefeng Yan and Yanbiao Niu and Lina Gong and Yanwen Guo and Mingqiang Wei}
}
```  

### Getting started


- Install PyTorch 1.6 or above and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  
### DEMore-Net Training and Test

- Using build_data.py/databuildMOR.py to  generate train.txt, gt.txt, and depth.txt.

- Train the DEMore-Net model:
```bash
python train.py
```
The checkpoints will be stored at `./ckpt`.

- Test the DEMore-Net model:
```bash
python infer.py
```
The test results will be saved to an html file here: `./ckpt/`.


### Acknowledgments
Our code is developed based on [DGNL-Net](https://github.com/xw-hu/DGNL-Net). We thank the awesome work provided by DGNL-Net.
And great thanks to the anonymous reviewers for their helpful feedback.

