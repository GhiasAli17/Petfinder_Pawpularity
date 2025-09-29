

import os
import random
import numpy as np
import pandas as pd
import cv2
import torch

from glob import glob
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.metrics import root_mean_squared_error as sklearn_rmse
from statsmodels.stats.outliers_influence import variance_inflation_factor

from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    CosineAnnealingLR
)




def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class MetricMonitor:
    """Tracks and computes the average of various metrics."""
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

def get_scheduler(optimizer, params):
    """Returns the learning rate scheduler based on configuration.
    Cosine Annealing adjusts the learning rate using a cosine function, which means the learning rate starts high, gradually decreases to a minimum
    """
    scheduler_params = params 
    scheduler_name = scheduler_params.get('schedulername')
    """ CosineAnnealingWarmRestarts is a variant that periodically restarts the learning rate back to the maximum, allowing the model to escape local minima.
    CosineAnnealingLR gradually decreases the learning rate from a maximum to a minimum value during training"""
    if scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params.get('T0', 5), 
            eta_min=scheduler_params.get('minlr', 1e-7),
            last_epoch=-1
        )
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('Tmax', 5),
            eta_min=scheduler_params.get('minlr', 1e-7),
            last_epoch=-1
        )
    else:
        scheduler = None
        
    return scheduler

def show_image(train_dataset, inline=4):
    """Displays a batch of random images from the dataset."""
    plt.figure(figsize=(20,10))
    for i in range(inline):
        rand = random.randint(0, len(train_dataset) - 1)
        image, _, label = train_dataset[rand]
        plt.subplot(1, inline, i % inline + 1)
        plt.axis('off')
        plt.imshow(image.permute(2, 1, 0)) 
        plt.title(f'Pawpularity: {label.item() * 100:.1f}')
    plt.show()

def usr_rmse_score(output, target):
    """Calculates the scaled RMSE score (for Pawpularity which is 0-100)."""
    with torch.no_grad():
        y_pred = torch.sigmoid(output) * 100
        target_scaled = target * 100 
        rmse = torch.sqrt(torch.mean((y_pred - target_scaled) ** 2))
    return rmse.item()


# ----********----  Figure, Images plot related function -----********-------

#1. Plotting Pawpularity distribution

def plotDistribution(train_df):
    sns.histplot(data=train_df, x='Pawpularity', bins=100)
    plt.axvline(train_df['Pawpularity'].mean(), c='red', ls='-', lw=3, label='Mean Pawpularity')
    plt.axvline(train_df['Pawpularity'].median(),c='blue',ls='-',lw=3, label='Median Pawpularity')
    plt.title('Distribution of Pawpularity Scores', fontsize=20, fontweight='bold')
    plt.legend()
    plt.show()


#2. Showing box plat and distribution across each feature
def plotDistributionAcrossEachFeature(features,train_df):
    for variable in features:
        fig, ax = plt.subplots(1,2,figsize=(10, 5))
        sns.boxplot(data=train_df, x=variable,hue=variable,palette=[ "#4c72b0", "#dd8452"], y='Pawpularity', ax=ax[0])
        sns.histplot(train_df, x="Pawpularity", hue=variable, kde=True, ax=ax[1])
        plt.suptitle(variable, fontsize=20, fontweight='bold')
        plt.show()


#3. Showing picture
def showPicture(num_of_pictures, train_jpg, train_df):
    fig, ax = plt.subplots(1, num_of_pictures, figsize=(10, 5))
    for i in range(num_of_pictures):
        img_path = train_jpg[i]
        img = plt.imread(img_path)
        #to get the img id
        img_id = Path(train_jpg[i]).stem
        # to retreive the record of only that id from pandas dataset
        score = train_df.loc[train_df["Id"] == img_id,"Pawpularity"].iloc[0]
        ax[i].imshow(img)
        ax[i].set_title(f"{img.shape} \n Pawpularity: {score}")
        ax[i].axis('off')

    plt.show()

 #4. Showing highest and lowest score pictures
def showPicture(num_of_pictures, train_jpg, train_df):
    fig, ax = plt.subplots(1, num_of_pictures, figsize=(10, 5))
    for i in range(num_of_pictures):
        img_path = train_jpg[i]
        img = plt.imread(img_path)
        #to get the img id
        img_id = Path(train_jpg[i]).stem
        # to retreive the record of only that id from pandas dataset
        score = train_df.loc[train_df["Id"] == img_id,"Pawpularity"].iloc[0]
        ax[i].imshow(img)
        ax[i].set_title(f"{img.shape} \n Pawpularity: {score}")
        ax[i].axis('off')

    plt.show()

#5. Plot correlation matrix
def plotCorrelationMatrix(train_df,figsize=(8,8)):
    # Slice the DataFrame to exclude 'Id' from correlation calculation
    correlation_data = train_df.loc[:, train_df.columns != 'Id']
    co_matrix = correlation_data.corr()
    mask = np.triu(np.ones_like(co_matrix, dtype=bool))
    # Plot the correlation matrix
    plt.figure(figsize=figsize)
    sns.heatmap(co_matrix, annot=True,mask=mask, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
    print(co_matrix["Pawpularity"][:-1])


#6. Plot VIF
def calculateVIF(train_df):
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = train_df.columns
    
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(train_df.values, i)
                            for i in range(len(train_df.columns))]  
    vif_data = vif_data.sort_values("VIF", ascending=False)
    return vif_data

#7. Extracting features from image

def extractFeature(train_df):
    features_list = []

    for idx, row in train_df.iterrows():
        path = row['path']
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Feature calculations
        brightness = np.mean(gray)
        contrast = np.std(gray)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        height, width = img.shape[:2]
        aspect_ratio = width / height

        features_list.append({
            "Id": row['Id'],  # we will use this for merging
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "sharpness": sharpness,
            "edge_density": edge_density,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio
        })

    return features_list

