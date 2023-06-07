import pandas as pd
import numpy as np
import utilities
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import csv
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from exploratory_analysis import generate_heat_map


def x_split(ov):

    training = pd.read_csv(f"../data/test_training_data/{ov}/final_num_training_data.csv", sep=':')
    test = pd.read_csv(f"../data/test_training_data/{ov}/final_num_test_data.csv", sep=':')
    x_train = training.drop(["School_Code"], axis=1)
    x_test = test.drop(["School_Code"], axis=1)
    x_train.to_csv(f"../data/test_training_data/{ov}/x_num_train.csv", sep=":", index=False)
    x_test.to_csv(f"../data/test_training_data/{ov}/x_num_test.csv", sep=":", index=False)
    # return x_train, x_test


def pca(ov):
    x_num_train = pd.read_csv(f"../data/test_training_data/{ov}/x_num_train.csv", sep=':')
    x_num_test = pd.read_csv(f"../data/test_training_data/{ov}/x_num_test.csv", sep=':')
    ss = StandardScaler().fit(x_num_train)
    input_train = ss.transform(x_num_train)
    input_test = ss.transform(x_num_test)
    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(input_train)
    x_test_pca = pca.transform(input_test)
    pd.DataFrame(x_train_pca).to_csv(f'../data/test_training_data/{ov}/x_pca_train.csv', index=False, header=True , sep=':')
    pd.DataFrame(x_test_pca).to_csv(f'../data/test_training_data/{ov}/x_pca_test.csv', index=False, header=True, sep=':')
    return pca


def generate_scree_plt(ov, pca):
    # print(pca)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component ' + ov)
    plt.title('Scree Plot')
    plt.savefig('../figures/PCA_OP/PCA_' + ov, dpi=400)
    plt.show()

def perform_pca():

    with open(utilities.output_variables_file_path) as f:
        output_variables = f.readlines()
        output_variables.remove("School_Code\n")
        output_variables.remove("Town\n")
        for ov in output_variables:
            ov = ov.replace("\n", "")
            x_split(ov) #generates x_num_final_train, x_num_final_test
            generate_heat_map(ov,'x_num_train')
            pca_ov=pca(ov)
            generate_scree_plt(ov,pca_ov)
            generate_heat_map(ov,'x_pca_train')
