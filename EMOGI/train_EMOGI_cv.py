# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:29:41 2018

@author: roman
"""

import argparse, os, sys
import tensorflow as tf
import utils, gcnIO, gcnPreprocessing
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import numpy as np
from train_EMOGI import *
from emogi import EMOGI
import gin

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Train EMOGI with cross-validation and save model to file')
    parser.add_argument('--config', help='Path to config.gin file',
                        type=str,
                        required=True
                        )
    args = parser.parse_args()
    return args



def single_cv_run(session, support, num_supports, features, y_train, y_test, train_mask, test_mask,
                  node_names, feature_names, args, model_dir):
    hidden_dims = [ x for x in args['hidden_dims']]
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
        'features': tf.placeholder(tf.float32, shape=features.shape, name='Features'),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='Labels'),
        'labels_mask': tf.placeholder(tf.int32, shape=train_mask.shape, name='LabelsMask'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='Dropout')
    }
    # construct model (including computation graph)
    model = EMOGI(placeholders=placeholders,
                  input_dim=features.shape[1],
                  learning_rate=args['lr'],
                  weight_decay=args['decay'],
                  num_hidden_layers=len(hidden_dims),
                  hidden_dims=hidden_dims,
                  pos_loss_multiplier=args['loss_mul'],
                  logging=True
    )
    # fit the model
    model = fit_model(model, session, features, placeholders,
                      support, args['epochs'], args['dropout'],
                      y_train, train_mask, y_test, test_mask,
                      model_dir)
    # Compute performance on test set
    performance_ops = model.get_performance_metrics()
    session.run(tf.local_variables_initializer())
    d = utils.construct_feed_dict(features, support, y_test,
                                  test_mask, placeholders)
    test_performance = session.run(performance_ops, feed_dict=d)
    print("Validataion/test set results:", "loss=", "{:.5f}".format(test_performance[0]),
        "accuracy=", "{:.5f}".format(
            test_performance[1]), "aupr=", "{:.5f}".format(test_performance[2]),
        "auroc=", "{:.5f}".format(test_performance[3]))

    # predict all nodes (result from algorithm)
    predictions = predict(session, model, features, support, y_test,
                          test_mask, placeholders)
    positive_indices_all_data_set = np.unique(np.concatenate((np.where(y_val > 0)[0], np.where(y_train > 0)[0], np.where(y_test > 0)[0]), axis = None))
    
    all_dataset_indices = np.concatenate((np.where(train_mask > 0)[0], np.where(test_mask > 0)[0], np.where(val_mask > 0)[0]), axis = None)

    predicted_positive_indices = np.where(predictions > 0.5)[0]
    pos_counter = 0
    for index in predicted_positive_indices:
        if index in positive_indices_all_data_set:
            pos_counter += 1 
    TPR = pos_counter/len(positive_indices_all_data_set)  
    print(f"TPR: {TPR}")
    predicted_negative_indices = np.where(predictions <= 0.5)[0]
    neg_counter = 0
    for index in predicted_negative_indices:
        if index in all_dataset_indices and index not in positive_indices_all_data_set:
            neg_counter += 1 
    TNR = neg_counter/(len(all_dataset_indices) - len(positive_indices_all_data_set))
    print(f"TNR: {TNR}")      
    gcnIO.save_predictions(model_dir, node_names, predictions)
    gcnIO.write_train_test_sets(model_dir, y_train, y_test, train_mask, test_mask)
    return test_performance

def run_all_cvs(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names,
                feature_names, args, output_dir):
    # preprocess features
    num_feat = features.shape[1]
    if num_feat > 1:
        print ("No row-normalization for 3D features!")
        #features = utils.preprocess_features(lil_matrix(features))
        #print ("Not row-normalizing...")
        #features = utils.sparse_to_tuple(lil_matrix(features))
    else:
        print("Not row-normalizing features because feature dim is {}".format(num_feat))
        #features = utils.sparse_to_tuple(lil_matrix(features))

    # get higher support matrices
    support, num_supports = utils.get_support_matrices(adj, args['support'])

    # construct splits for k-fold CV
    y_all = np.logical_or(y_train, y_val)
    mask_all = np.logical_or(train_mask, val_mask)
    k_sets = gcnPreprocessing.cross_validation_sets(y=y_all,
                                                    mask=mask_all,
                                                    folds=args['cv_runs']
    )

    performance_measures = []
    for cv_run in range(args['cv_runs']):
        model_dir = os.path.join(output_dir, 'cv_{}'.format(cv_run))
        y_tr, y_te, tr_mask, te_mask = k_sets[cv_run]
        with tf.Session() as sess:
            val_performance = single_cv_run(sess, support, num_supports, features, y_tr,
                                            y_te, tr_mask, te_mask, node_names, feature_names,
                                            args, model_dir)
            performance_measures.append(val_performance)
        tf.reset_default_graph()
    # save hyper Parameters
    data_rel_to_model = os.path.relpath(args['data'], output_dir)
    args['data'] = data_rel_to_model
    gcnIO.write_hyper_params(args, args['data'], os.path.join(output_dir, 'hyper_params.txt'))
    return performance_measures

@gin.configurable
def create_args_dict(hidden_dims :list, loss_mul :float, data :str, cv_runs: int, epochs: int):
    args_dict = {'lr': 0.001, 'support': 1, 'decay': 0.05, 'dropout': 0.5}
    args_dict.update({'epochs': epochs})
    args_dict.update({'hidden_dims' : hidden_dims})
    args_dict.update({'loss_mul' : loss_mul})
    args_dict.update({'data': data})
    args_dict.update({'cv_runs': cv_runs})
    return args_dict

if __name__ == "__main__":
    # load config file
    args = parse_args()
    config_file_path = args.config
    gin.parse_config_file(config_file_path)
    output_dir = gcnIO.create_model_dir()
    args_dict = create_args_dict()
    
    # load data and preprocess it
    input_data_path = args_dict['data']
    data = gcnIO.load_hdf_data(input_data_path, feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data
    print("Read data from: {}".format(input_data_path))
    
    print (f"args_dict: {args_dict}")
    performance_measures = run_all_cvs(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,
                node_names, feature_names, args_dict, output_dir)

    avg_auroc = 0
    avg_auprc = 0
    for item in performance_measures:
        avg_auprc += item[2]
        avg_auroc += item[3]
    avg_auroc /= args_dict['cv_runs']    
    avg_auprc /= args_dict['cv_runs']
    print(f"performance_measures: {performance_measures} auroc: {avg_auroc}, auprc: {avg_auprc}")

    config_dir, _ = os.path.split(config_file_path)
    with open(os.path.join(config_dir, 'result'), 'w') as f:
        f.write(f'{-avg_auroc}\n')    