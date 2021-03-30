![Python >=3.5](https://img.shields.io/badge/Python->=3.5-yellow.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)

# TransReID: Transformer-based Object Re-Identification [[arxiv]](https://arxiv.org/abs/2102.04378)

The *official* repository for  [TransReID: Transformer-based Object Re-Identification](https://arxiv.org/abs/2102.04378) achieves state-of-the-art performances on object re-ID, including person re-ID and vehicle re-ID.

## Pipeline

![framework](figs/framework.png)

## Abaltion Study of Transformer-based Strong Baseline

![framework](figs/ablation.png)



## Requirements

### Installation

```bash
pip install -r requirements.txt
(we use /torch 1.6.0 /torchvision 0.7.0 /timm 0.3.2 /cuda 10.1 / 16G or 32G V100 for training and evaluation.
Note that we use torch.cuda.amp to accelerate speed of training which requires pytorch >=1.6)
```

### Prepare Datasets

```bash
mkdir data
```

Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775),[Occluded-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset), and the vehicle datasets [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html), [VeRi-776](https://github.com/JDAI-CV/VeRidataset), 
Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── images ..
├── MSMT17
│   └── images ..
├── dukemtmcreid
│   └── images ..
├── Occluded_Duke
│   └── images ..
├── VehicleID_V1.0
│   └── images ..
└── VeRi
    └── images ..
```

### Prepare DeiT or ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth), [ViT-Small](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth), [DeiT-Small](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth), [DeiT-Base](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth)

## Training

We utilize 1  GPU for training.

```bash
python train.py --config_file configs/transformer_base.yml MODEL.DEVICE_ID "('your device id')" MODEL.STRIDE_SIZE ${1} MODEL.SIE_CAMERA ${2} MODEL.SIE_VIEW ${3} MODEL.JPM ${4} MODEL.TRANSFORMER_TYPE ${5} OUTPUT_DIR ${OUTPUT_DIR} DATASETS.NAMES "('your dataset name')"
```

#### Arguments

- `${1}`: stride size for pure transformer, e.g. [16, 16], [14, 14], [12, 12]
- `${2}`: whether using SIE with camera, True or False.
- `${3}`: whether using SIE with view, True or False.
- `${4}`: whether using JPM, True or False.
- `${5}`: choose transformer type from `'vit_base_patch16_224_TransReID'`,(The structure of the deit is the same as that of the vit, and only need to change the imagenet pretrained model)  `'vit_small_patch16_224_TransReID'`,`'deit_small_patch16_224_TransReID'`,
- `${OUTPUT_DIR}`: folder for saving logs and checkpoints, e.g. `../logs/market1501`

**or you can directly train with following  yml and commands:**

```bash
# DukeMTMC transformer-based baseline
python train.py --config_file configs/DukeMTMC/vit_base.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC baseline + JPM
python train.py --config_file configs/DukeMTMC/vit_jpm.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC baseline + SIE
python train.py --config_file configs/DukeMTMC/vit_sie.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC TransReID (baseline + SIE + JPM)
python train.py --config_file configs/DukeMTMC/vit_transreid.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC TransReID with stride size [12, 12]
python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"

# MSMT17
python train.py --config_file configs/MSMT17/vit_transreid.yml MODEL.DEVICE_ID "('0')"
# OCC_Duke
python train.py --config_file configs/OCC_Duke/vit_transreid.yml MODEL.DEVICE_ID "('0')"
# Market
python train.py --config_file configs/Market/vit_transreid.yml MODEL.DEVICE_ID "('0')"

```

## Evaluation

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

**Some examples:**

```bash
# DukeMTMC
python test.py --config_file configs/DukeMTMC/vit_transreid.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT '../logs/duke_vit_transreid/transformer_120.pth'
# MSMT17
python test.py --config_file configs/MSMT17/vit_transreid.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT '../logs/msmt17_vit_transreid/transformer_120.pth'
# OCC_Duke
python test.py --config_file configs/OCC_Duke/vit_transreid.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT '../logs/occ_duke_vit_transreid/transformer_120.pth'
# Market
python test.py --config_file configs/Market/vit_transreid.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT '../logs/market_vit_transreid/transformer_120.pth'

```

## Trained Models and logs (Size 256)

![framework](figs/sota.png)

|       Model       |                            MSMT17                            |                            Market                            |                             Duke                             |                           OCC_Duke                           | VeRi | VehicleID |
| :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--: | --------- |
|   baseline(ViT)   | [model](https://drive.google.com/file/d/1iF5JNPw9xi-rLY3Ri9EY-PFAkK6Vg_Pf/view?usp=sharing) \| [log](https://drive.google.com/file/d/1oCnLpwv-V_RU7_BNXFsIgXKxAm2QAD7n/view?usp=sharing) | [model](https://drive.google.com/file/d/1crYsKRrW4eUq6abT4KK8_atMLFsbq56W/view?usp=sharing) \| [log](https://drive.google.com/file/d/1YSo6FgJ42SOv3TTQvzE_4V1r3Ma608lZ/view?usp=sharing) | [model](https://drive.google.com/file/d/17GQqFuTleAZWLD92AtEd1c_dnTyZHl4k/view?usp=sharing) \| [log](https://drive.google.com/file/d/1a8Ci3qN4Y47LRWqgbeF4HJON1hBmeLCn/view?usp=sharing) | [model](https://drive.google.com/file/d/1uHX5j7yepalN1EINdF9lzrT3iDWj-pr9/view?usp=sharing) \| [log](https://drive.google.com/file/d/1urUfrvML_7qKvqXyz6Yl4msJS6nTNbe5/view?usp=sharing) | TBD  | TBD       |
| TransReID^*(ViT)  | [model](https://drive.google.com/file/d/1x6Na97ycxS0t2Dn_0iRKWe1U5ccIqASK/view?usp=sharing) \| [log](https://drive.google.com/file/d/14TPDaU2T0WLTsg0iEHJFnqwzSTrpzC0B/view?usp=sharing) | [model](https://drive.google.com/file/d/11p4RjmpCGGAS-876VEt7OoFrUeHTUlyO/view?usp=sharing) \| [log](https://drive.google.com/file/d/1SWNtnhEVoDu3Uixf5XBCQlvXYapVrk7w/view?usp=sharing) | [model](https://drive.google.com/file/d/1BipxoqyThefQviJzuJIKtFJvNblIlPGN/view?usp=sharing) \| [log](https://drive.google.com/file/d/11dE_kbNWbvmo-3qUShN7qsrTsqd89Eoc/view?usp=sharing) | [model](https://drive.google.com/file/d/1VJg4rTA43TCHkR9hTIBu8S2Sy1KiTnSJ/view?usp=sharing) \| [log](https://drive.google.com/file/d/1I1xTSBl1v-QBSyxxAB7xIszW_fu9oT6g/view?usp=sharing) | TBD  | TBD       |
| TransReID^*(DeiT) |                         model \| log                         | [model](https://drive.google.com/file/d/1cbUK2KozdPSoewzvF0ucFQnZ0yfZiu_H/view?usp=sharing) \| [log](https://drive.google.com/file/d/1C9glb0kc5thfU3U9Yrr6z7h5oYgMwHfy/view?usp=sharing) | [model](https://drive.google.com/file/d/1ltaX9zGFO31Wwwu47K9c4WTTBZVLdzLw/view?usp=sharing) \| [log](https://drive.google.com/file/d/13H9usPg7pG5b6Eglx0EiKDiU6n3chBnT/view?usp=sharing) |                         model \| log                         | TBD  | TBD       |

（We reorganize the code. Now we are running each experiment one by one to make sure that you can reproduce the results in our paper. We will gradually upload the logs and models after training in the following weeks. Thanks for your attention.）

## Acknowledgement

Codebase from [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) , [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

## Citation

If you find this code useful for your research, please cite our paper

```
@article{he2021transreid,
  title={TransReID: Transformer-based Object Re-Identification},
  author={He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
  journal={arXiv preprint arXiv:2102.04378},
  year={2021}
}
```

## Contact

If you have any question, please feel free to contact us. E-mail: [shuting_he@zju.edu.cn](mailto:shuting_he@zju.edu.cn) , [haoluocsc@zju.edu.cn](mailto:haoluocsc@zju.edu.cn)

