from random import sample
import numpy as np
import pandas as pd
import networkx as nx
import sys, os, h5py
sys.path.append(os.path.abspath('../EMOGI'))
import gcnPreprocessing
import gcnIO
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA



def do_pca_positives_vs_negatives(y_train, y_val, y_test, train_mask, val_mask, test_mask, feat_names, is_pca, is_scatter):
    
    positive_indices = np.unique(np.concatenate((np.where(y_val > 0.1)[0], np.where(y_train > 0.1)[0], np.where(y_test > 0.1)[0]), axis = None))
    all_dataset_indices = np.unique(np.concatenate((np.where(train_mask > 0)[0], np.where(test_mask > 0)[0], np.where(val_mask > 0)[0]), axis = None)) 
    negative_labelled_indices = []
    for index in all_dataset_indices:
        if index not in positive_indices:
            negative_labelled_indices.append(index)
            
    _, n_features = np.shape(features)
    positives_feature_vectors = [[features[i][k] for k in np.arange(n_features)] for i in positive_indices]
    negatives_feature_vectors = [[features[i][k] for k in np.arange(n_features)] for i in negative_labelled_indices]       
    compute_pca_and_plot(positives_feature_vectors, negatives_feature_vectors, feat_names, is_pca, is_scatter)  
    
def compute_pca_and_plot(positive_features, negative_features, feat_names, is_pca :bool, is_scatter :bool):
    if is_pca:
        positive_features = [[features[i] for i in [0, 1]] for features in positive_features]
        negative_features = [[features[i] for i in [0, 1]] for features in negative_features]
        pca = PCA(n_components=2)
        host_factors_principal_components = pca.fit_transform(positive_features)
        df_pca_host_factors = pd.DataFrame(data = host_factors_principal_components
                    , columns = ['principal component 1', 'principal component 2'])    
        
        negative_genes_principal_components = pca.fit_transform(negative_features)
        df_pca_negative_genes = pd.DataFrame(data = negative_genes_principal_components
                    , columns = ['principal component 1', 'principal component 2'])    
        
        fig = plt.figure(figsize = (16,16))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 Component PCA', fontsize = 20)

        ax.scatter(df_pca_negative_genes['principal component 1']
                    , df_pca_negative_genes['principal component 2']
                    , c = "b"
                    , alpha = 0.3
                    , s = 10)
        ax.scatter(df_pca_host_factors['principal component 1']
                    , df_pca_host_factors['principal component 2']
                    , c = "r"
                    , alpha = 0.9
                    , s = 20)
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.legend(["set of the potential negatives", "positive set samples"], fontsize = 12)
        ax.grid()
        plt.show()    

        return
    elif is_scatter:
        positive_features = [[features[i] for i in [0, 1]] for features in positive_features]
        negative_features = [[features[i] for i in [0, 1]] for features in negative_features]
        feat_0 = str(feat_names[0])[2:-1]
        feat_1 = str(feat_names[1])[2:-1]
        df_host_factors = pd.DataFrame(data = positive_features
                    , columns = [feat_0, feat_1])    
        
        df_negative_genes = pd.DataFrame(data = negative_features
                    , columns = [feat_0, feat_1])    
        
        fig = plt.figure(figsize = (16,16))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(feat_0, fontsize = 15)
        ax.set_ylabel(feat_1, fontsize = 15)
        ax.set_title('Scatter Plot', fontsize = 20)

        ax.scatter(df_negative_genes[feat_0]
                    , df_negative_genes[feat_1]
                    , c = "b"
                    , alpha = 0.3
                    , s = 10)
        ax.scatter(df_host_factors[feat_0]
                    , df_host_factors[feat_1]
                    , c = "r"
                    , alpha = 0.9
                    , s = 20)
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.legend(["set of the potential negatives", "positive set samples"], fontsize = 12)
        ax.grid()
        plt.show()  
        return       
    
    positive_features = [features[1] for features in positive_features]
    negative_features = [features[1] for features in negative_features]
    
    print(f" median positive_features { np.median(positive_features)} std positive_features {np.std(positive_features)}")
    print(f" median negative_features { np.median(negative_features)} std negative_features {np.std(negative_features)}")
    
    bins = np.linspace(-2.5, 2.5, 70)

    fig = plt.gcf()
    fig.set_size_inches(22, 12)

    plt.title(str(feat_names[1])[2:-1])
    plt.hist(positive_features, bins, alpha=0.5, density=True, label=f'positive_features -> median; { round(np.median(positive_features),3)}, std:  {round(np.std(positive_features), 3)}')
    plt.hist(negative_features, bins, alpha=0.5, density=True, label=f'negative_features -> median; { round(np.median(negative_features),3)}, std:  {round(np.std(negative_features), 3)}')
    plt.legend(loc='upper right')
    plt.ylabel('Counts')
    plt.xlabel('Feature value')
    plt.show()

    fig.savefig('Proteomics_at_24H_signed_extended.png', dpi=100)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Data preprocessing for transcriptome and proteome from cell lines')

    parser.add_argument('--input_data_path', help='hdf5 file path',
                    type=str
                    )
    parser.add_argument('--is_pca', action='store_true', default = False, help = "Plot PCA")
    parser.add_argument('--is_scatter', action='store_true', default = False, help = "Do scatter plot")
    
    args = parser.parse_args()
    input_data_path = args.input_data_path
    is_pca = args.is_pca
    is_scatter = args.is_scatter
    
    data = gcnIO.load_hdf_data(input_data_path)
    network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data

    print(f"features dim: {np.shape(features)}")
    print(f"labels dim: {np.shape(y_train)}")

    print(f"feature names {feat_names}")

    print(f"number of positives {np.sum(y_train)}")
    print(f"number of negatives {np.sum(train_mask) - np.sum(y_train)}")

    print(f"number of positive training labels {np.sum(y_train)}")
    print(f"number of negative training labels {np.sum(train_mask) - np.sum(y_train)}")
    print(f"number of positive test labels {np.sum(y_test)}")
    print(f"number of negative test labels {np.sum(test_mask) - np.sum(y_test)}")
    print(f"number of positive val labels {np.sum(y_val)}")
    print(f"number of negative val labels {np.sum(val_mask) - np.sum(y_val)}")  
    
    do_pca_positives_vs_negatives(y_train, y_val, y_test, train_mask, val_mask, test_mask, feat_names, is_pca, is_scatter)
