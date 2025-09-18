
import random
import numpy as np
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


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
def plotCorrelationMatrix(train_df):
    # Slice the DataFrame to exclude 'Id' from correlation calculation
    correlation_data = train_df.loc[:, train_df.columns != 'Id']
    co_matrix = correlation_data.corr()
    # Plot the correlation matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(co_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
    print(co_matrix["Pawpularity"][:-1])
