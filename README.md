# Distribution-Aware and Class-Adaptive Aggregation for Few-Shot Hyperspectral Image Classification【TGRS2024】

Code for the paper: [Distribution-Aware and Class-Adaptive Aggregation for Few-Shot Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/10606525).

<p align='center'>
  <img src='Figures/DA-CAA.png' width="800px">
</p>

## Abstract:
Recently, few-shot learning based on meta-learning has shown great potential in hyperspectral image classification (HSIC) due to its excellent adaptability to limited training samples. Despite achieving promising results, the existing methods ignore the interaction between the source domain (with abundant-labeled base-class samples) and the target domain (with few-labeled novel-class samples), as well as between the support set and the query set. This issue makes the resulting model usually biased toward the source domain and not robust to the sample variance of novel classes, posing a bottleneck to the improvement of HSIC performance. To overcome these limitations, we propose a flexible and effective distribution-aware and class-adaptive aggregation (DA-CAA) method for few-shot HSIC by transferring the class-level distribution information learned from the base classes to the novel classes. Specifically, we first employ a variational autoencoder (VAE), which is pretrained on abundant-labeled base-class samples, to encode the support set samples as class distributions. Subsequently, we sample class-level features from the learned distribution and adaptively aggregate them with sample-specific query features. This operation not only enhances cross-domain information interaction in a distribution-learning manner, but also ensures that the aggregated features across classes inherit both class-level and sample-specific information. Our proposed class-adaptive aggregation (CAA) encourages complementary fusion of features from all classes, which is beneficial for reducing class confusion. Experiments on four benchmark datasets demonstrate the effectiveness and flexibility of our approach.

## Citation

if you find the code useful for your research, please kindly cite:

[1] Y. Guo, B. Fan, Y. Feng, X. Jia and M. He, "Distribution-Aware and Class-Adaptive Aggregation for Few-Shot Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-16, 2024, Art no. 5525216, doi: 10.1109/TGRS.2024.3432734.

```
@ARTICLE{10606525,
  author={Guo, Ying and Fan, Bin and Feng, Yan and Jia, Xiuping and He, Mingyi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Distribution-Aware and Class-Adaptive Aggregation for Few-Shot Hyperspectral Image Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Training;Feature extraction;Task analysis;Metalearning;Adaptation models;Image classification;Few shot learning;Feature aggregation;few-shot learning;hyperspectral image classification (HSIC);variational autoencoder (VAE)},
  doi={10.1109/TGRS.2024.3432734}}

```

## Requirements

CUDA Version: 11.6

torch: 1.13.0

Python: 3.9

## Datasets

```
├── Patch9_TRIAN_META_DATA_imdb_ocbs.pickle
├── test_ocbs
│   ├── PaviaU_data.mat
│   ├── PaviaU_gt.mat
└── train_ocbs
    ├── Botswana_data.mat
    ├── Botswana_gt.mat
    ├── Chikusei_data.mat
    ├── Chikusei_gt.mat
    ├── KSC_data.mat
    └── KSC_gt.mat
```
1) Please prepare the training and test data as operated in the paper. The used OCBS band selection method is referred to https://github.com/tanmlh.
2) Run "trainMetaDataProcess.py" to generate the meta-training data "Patch9_TRIAN_META_DATA_imdb_ocbs.pickle". And you can choose to download the meta-training data through Baidu Netdisk (link:https://pan.baidu.com/s/1vSDJTe32GQg-fwkaJ59DUw Extraction code:pj1n ) 
3) Run "python train.py".

## Acknowledgements

This project would not have been possible without relying on some awesome repos: [DCFSL](https://github.com/Li-ZK/DCFSL-2021), [CMFSL](https://github.com/B-Xi/TIP_2022_CMFSL) and [VAE](https://github.com/AntixK/PyTorch-VAE). We thank the original authors for their excellent work.

Licensing
--
Copyright (C) 2024 Ying Guo

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
