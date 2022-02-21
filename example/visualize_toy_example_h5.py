from random import sample
import numpy as np
import pandas as pd
import networkx as nx
import sys, os, h5py
sys.path.append(os.path.abspath('../EMOGI'))
import gcnPreprocessing
import gcnIO
import matplotlib.pyplot as plt


data = gcnIO.load_hdf_data('toy_example.h5')
network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
features_df = pd.DataFrame(features, index=node_names[:, 1], columns=feat_names)

print(f"features dim: {np.shape(features)}")
print(f"labels dim: {np.shape(y_train)}")

print(f"feature names {feat_names}")

print(f"Mutation frequency per cancer type MAX: {np.max(features[:,0:15])}")
print(f"Mutation frequency per cancer type MIN: {np.mean(features[:,0:15])}")
print(f"Methilation per cancer type MAX: {np.max(features[:,16:31])}")
print(f"Methilation per cancer type MIN: {np.min(features[:,16:31])}")
print(f"Gene expression per cancer type MAX: {np.max(features[:,32:47])}")
print(f"Gene expression per cancer type MIN: {np.min(features[:,32:47])}")
sample = "b'KRTAP23-1'"
print(f"node names: {type(sample)}")