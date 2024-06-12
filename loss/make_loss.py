# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    print("Configuring loss function...")
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    print(f"Center loss configured with num_classes={num_classes} and feat_dim={feat_dim}")
    
    # in the case of python train.py --config_file configs/Market/vit_jpm.yml MODEL.DEVICE_ID "('0')" 
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            # This is initialized!
            triplet = TripletLoss()
            print("Using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print(f"Using triplet loss with margin: {cfg.SOLVER.MARGIN}")
    else:
        print(f"Expected METRIC_LOSS_TYPE should be 'triplet' but got {cfg.MODEL.METRIC_LOSS_TYPE}")

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print(f"Label smoothing is on. Number of classes: {num_classes}")

    if sampler == 'softmax':
        print("Sampler is softmax. Using cross entropy loss.")
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        print("Sampler is softmax_triplet. Using combined softmax and triplet loss.")
        def loss_func(score, feat, target, target_cam):
            print("Calculating loss...")
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        print("Score is a list. Calculating ID loss with label smoothing.")
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        print("Score is not a list. Calculating ID loss with label smoothing.")
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        print("Feat is a list. Calculating triplet loss.")
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        print("Feat is not a list. Calculating triplet loss.")
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    print(f"Loss calculated: ID_LOSS={ID_LOSS}, TRI_LOSS={TRI_LOSS}, Total Loss={loss}")
                    return loss
                else:
                    if isinstance(score, list):
                        print("Score is a list. Calculating ID loss without label smoothing.")
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        print("Score is not a list. Calculating ID loss without label smoothing.")
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        print("Feat is a list. Calculating triplet loss.")
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        print("Feat is not a list. Calculating triplet loss.")
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    print(f"Loss calculated: ID_LOSS={ID_LOSS}, TRI_LOSS={TRI_LOSS}, Total Loss={loss}")
                    return loss
            else:
                print(f"Expected METRIC_LOSS_TYPE should be 'triplet' but got {cfg.MODEL.METRIC_LOSS_TYPE}")

    else:
        print(f"Expected sampler should be 'softmax', 'triplet', 'softmax_triplet' or 'softmax_triplet_center' but got {cfg.DATALOADER.SAMPLER}")

    print("Loss function configuration complete.")
    return loss_func, center_criterion
