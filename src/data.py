import numpy as np
import pandas as pd
import torch
import albumentations
from albumentations.pytorch import ToTensorV2
from sklearn import model_selection
import cv2
from torch.utils.data import Dataset


def get_config():
    """Returns the main configuration dictionary."""
    return {
        'seed': 42,
        'debug': False,
        'numfolds': 5,
    }

def get_params(features):
    """Returns the model and training parameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return {
        'model': 'vit_large_patch32_384',
        'densefeatures': features,
        'pretrained': True,
        'inpchannels': 3,
        'imsize': 384,
        'device': device,
        'lr': 1e-5,
        'weight_decay': 1e-6,
        'batchsize': 32,
        'numworkers': 0,
        'epochs': 10,
        'outfeatures': 1,
        'dropout': 0.2,
        'numfold': 5, 
        'mixup': False,
        'mixupalpha': 1.0,
        'schedulername': 'CosineAnnealingWarmRestarts',
        'T0': 5,
        'Tmax': 5,
        'Tmult': 1,
        'minlr': 1e-7,
        'max_lr': 1e-4
    }

def createfolds(data, num_splits, seed):
    """
    Splits the dataframe into stratified folds based on 'Pawpularity'.
    """
    data["kfold"] = -1
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, "bins"] = pd.cut(data["Pawpularity"], bins=num_bins, labels=False)
    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    data = data.drop("bins", axis=1)
    return data

import albumentations
from albumentations.pytorch import ToTensorV2

def get_train_transforms(dim):
    """
    Returns the training data augmentation and normalization pipeline.
    """
    return albumentations.Compose(
        [
            albumentations.Resize(dim, dim),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
     
            albumentations.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=45, 
                p=0.5
            ),

            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2,
                val_shift_limit=0.2, 
                p=0.5
            ),
        
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms(dim):
    """
    Returns the validation data normalization pipeline.

    """
    return albumentations.Compose(
        [
            albumentations.Resize(dim, dim),
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0)
        ]
    )

class CuteDataset(Dataset):
    """
    Custom Dataset for loading images and dense features.
    """
    def __init__(self, images_filepaths, dense_features, targets, transform=None):
        self.images_filepaths = images_filepaths
        self.dense_features = dense_features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        dense = self.dense_features[idx, :]
        label = torch.tensor(self.targets[idx]).float() 
        return image, dense, label
    
