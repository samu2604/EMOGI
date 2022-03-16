from statistics import mean, stdev, median
import pandas as pd
import numpy as np
import argparse
import os.path
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random  
import h5py 

def create_hdf5_container(network, features, node_names, feat_names, y_train, train_mask, y_test, test_mask, y_val, val_mask, container_name :str):
    f = h5py.File(container_name + '.h5', 'w')
    string_dt = h5py.special_dtype(vlen=str)
    f.create_dataset('network', data=network, shape=network.shape)
    f.create_dataset('features', data=features, shape=features.shape)
    f.create_dataset('gene_names', data=node_names, dtype=string_dt)
    f.create_dataset('y_train', data=y_train, shape=y_train.shape)
    f.create_dataset('y_val', data=y_val, shape=y_val.shape)
    f.create_dataset('y_test', data=y_test, shape=y_test.shape)
    f.create_dataset('mask_train', data=train_mask, shape=train_mask.shape)
    f.create_dataset('mask_val', data=val_mask, shape=val_mask.shape)
    f.create_dataset('mask_test', data=test_mask, shape=test_mask.shape)
    f.create_dataset('feature_names', data=feat_names, dtype=string_dt)
    f.create_dataset('features_raw', data=features, shape=features.shape)
    f.close()

def create_adjacency_matrix_and_feature_matrix(ppi_file_path :str, feature_file_path :str):
    df_string_ppi = pd.read_csv(ppi_file_path)
    json_file = open(feature_file_path, "r")
    features_dict = json.load(json_file)    
    gene_list = [gene_name for gene_name in features_dict.keys()]
    list_of_hub_genes = df_string_ppi["gene_name_1"].unique()
    
    if not os.path.isfile("network.csv") and not os.path.isfile("features.csv") and not os.path.isfile("randomized_gene_list.txt"):
        print("Created randomized gene list")
        gene_indeces_list =  np.random.permutation(np.arange(0, len(gene_list)))
        randomized_gene_list = [gene_list[index] for index in gene_indeces_list]   
        np.savetxt("randomized_gene_list.txt", randomized_gene_list, delimiter = " ", fmt="%s")
          
        network = np.zeros([len(randomized_gene_list) ,len(randomized_gene_list)])
        print("Adjacency matrix creation")
        for gene in list_of_hub_genes:
            genes_connected_to_gene = df_string_ppi[df_string_ppi["gene_name_1"] == gene]["gene_name_2"].values
            index_gene = randomized_gene_list.index(gene)
            for gene_connected in genes_connected_to_gene:          
                index_gene_connected_to_gene = randomized_gene_list.index(gene_connected)
                if network[index_gene, index_gene_connected_to_gene] == 0 and index_gene != index_gene_connected_to_gene: 
                    network[index_gene, index_gene_connected_to_gene] = 1 # here we can use a value proportional to the interaction strength 
                    network[index_gene_connected_to_gene, index_gene] = 1 # here we can use a value proportional to the interaction strength
        np.savetxt("network.csv", network, delimiter = ",") 

        features = np.zeros([len(randomized_gene_list) ,2])
        print("Feature matrix creation")
        for gene_index, gene in enumerate(randomized_gene_list):
            features[gene_index, 0] = features_dict[gene][2] # transcriptomics at 24h
            features[gene_index, 1] = features_dict[gene][5] # proteomics at 24h  
        np.savetxt("features.csv", features, delimiter = ",")                 
    else:
        network =  np.loadtxt(open("network.csv"), delimiter = ",") 
        features = np.loadtxt(open("features.csv"), delimiter = ",")  
        randomized_gene_list = np.genfromtxt("randomized_gene_list.txt", dtype=str)    

    node_names = []
    for gene_name in randomized_gene_list:
        node_names.append([gene_name, gene_name])
    
    node_names = np.array(node_names, dtype=object)    
    feat_names = np.array(["transcriptomics_24h", "proteomics_24"], dtype=object)
                     
    return network, features, node_names, feat_names
    
def compute_pca_and_plot(positive_features, negative_features):
    # positive_features = [[features[i] for i in [2, 5]] for features in positive_features]
    # negative_features = [[features[i] for i in [2, 5]] for features in negative_features]
    # pca = PCA(n_components=2)
    # host_factors_principal_components = pca.fit_transform(positive_features)
    # df_pca_host_factors = pd.DataFrame(data = host_factors_principal_components
    #              , columns = ['principal component 1', 'principal component 2'])    
    
    # negative_genes_principal_components = pca.fit_transform(negative_features)
    # df_pca_negative_genes = pd.DataFrame(data = negative_genes_principal_components
    #              , columns = ['principal component 1', 'principal component 2'])    
    
    # fig = plt.figure(figsize = (16,16))
    # ax = fig.add_subplot(1,1,1) 
    # ax.set_xlabel('Principal Component 1', fontsize = 15)
    # ax.set_ylabel('Principal Component 2', fontsize = 15)
    # ax.set_title('2 Component PCA', fontsize = 20)

    # ax.scatter(df_pca_negative_genes['principal component 1']
    #                , df_pca_negative_genes['principal component 2']
    #                , c = "b"
    #                , alpha = 0.3
    #                , s = 10)
    # ax.scatter(df_pca_host_factors['principal component 1']
    #                , df_pca_host_factors['principal component 2']
    #                , c = "r"
    #                , alpha = 0.9
    #                , s = 20)
    # ax.set_xlim([-6, 6])
    # ax.set_ylim([-6, 6])
    # ax.legend(["set of the potential negatives", "positive set samples"], fontsize = 12)
    # ax.grid()
    
    positive_features = [features[1] for features in positive_features]
    negative_features = [features[1] for features in negative_features]
    
    print(f" mean value positive_features { mean(positive_features)} std positive_features {stdev(positive_features)}")
    print(f" mean value negative_features { mean(negative_features)} std negative_features {stdev(negative_features)}")
    
    bins = np.linspace(-1, 1, 60)

    fig = plt.gcf()
    fig.set_size_inches(22, 12)

    plt.title("Transcriptomics at 12H")
    plt.hist(positive_features, bins, alpha=0.5, density=True, label=f'positive_features -> mean value; { round(mean(positive_features),3)}, std:  {round(stdev(positive_features), 3)}')
    plt.hist(negative_features, bins, alpha=0.5, density=True, label=f'negative_features -> mean value; { round(mean(negative_features),3)}, std:  {round(stdev(negative_features), 3)}')
    plt.legend(loc='upper right')
    plt.ylabel('Counts')
    plt.xlabel('Feature value')
    plt.show()

    fig.savefig('Transcriptomics_at_12H.png', dpi=100)
    
    plt.show()
    
def sample_from_list_and_return_remaining_list(list_to_sample, number_elements_to_sample):
    if number_elements_to_sample > len(list_to_sample):
        print("Number of elements to sample is bigger then the sampling list")
    sampled_elements = random.sample(list_to_sample, number_elements_to_sample)
    for element in sampled_elements:
       list_to_sample.remove(element)
                
    return sampled_elements, list_to_sample
    
def train_test_val_sampler(list_to_sample, number_train_samples, number_test_samples, number_val_samples):
    training_samples, list_to_sample = sample_from_list_and_return_remaining_list(list_to_sample, number_train_samples)
    test_samples, list_to_sample = sample_from_list_and_return_remaining_list(list_to_sample, number_test_samples)
    validation_samples, list_to_sample = sample_from_list_and_return_remaining_list(list_to_sample, number_val_samples)
    return training_samples, test_samples, validation_samples
        
    
def train_test_val_split(host_factor_data_path, strong_host_factors):
    node_gene_list = np.genfromtxt("randomized_gene_list.txt", dtype=str)
    nodes_number = len(node_gene_list)
    y_train = np.zeros([nodes_number, 1], float)
    y_test = np.zeros([nodes_number, 1], float)
    y_val = np.zeros([nodes_number, 1], float)
    train_mask = np.zeros(nodes_number, float)
    test_mask = np.zeros(nodes_number, float)
    val_mask = np.zeros(nodes_number, float)
    
    # Read positive labels
    positive_host_factors_file = open(strong_host_factors, "r")
    positive_host_factors = []
    for line in positive_host_factors_file:
        gene = line.strip()
        positive_host_factors.append(gene)
    
    host_factors = pd.ExcelFile(host_factor_data_path)
    host_factors_dfs = {sheet_name: host_factors.parse(sheet_name) for sheet_name in host_factors.sheet_names}
    potential_host_factors_to_remove = host_factors_dfs["host_factors"]["Gene name"].unique()    
    
    # select and split positive labels in my dataset
    positive_labels = []
    for gene in positive_host_factors:
        if gene in node_gene_list:
            positive_labels.append(gene)
    
    number_positives = len(positive_labels)
    train_set_pos = int(number_positives * 0.65)
    test_set_pos = int(number_positives * 0.25)
    val_set_pos = int(number_positives * 0.1) + (number_positives - test_set_pos - train_set_pos - int(number_positives * 0.1))       
    
    # select the potential negatives in my dataset
    potential_negative_labels = []     
    for gene in node_gene_list:
        if gene not in potential_host_factors_to_remove and gene not in positive_labels:
            potential_negative_labels.append(gene)
            
    training_positive_samples, test_positive_samples, validation_positive_samples = train_test_val_sampler(positive_labels, train_set_pos, test_set_pos, val_set_pos)    
    training_negative_samples, test_negative_samples, validation_negative_samples = train_test_val_sampler(potential_negative_labels, 3*train_set_pos, 3*test_set_pos, 3*val_set_pos)    
    
    fill_labels_and_mask(training_positive_samples, training_negative_samples ,node_gene_list, y_train, train_mask)
    fill_labels_and_mask(test_positive_samples, test_negative_samples ,node_gene_list, y_test, test_mask)
    fill_labels_and_mask(validation_positive_samples, validation_negative_samples ,node_gene_list, y_val, val_mask)  
    
    # print(f" y_train {sum(y_train)} {len(training_positive_samples)}")
    # print(f" y_test {sum(y_test)} {len(test_positive_samples)}")
    # print(f" y_val {sum(y_val)} {len(validation_positive_samples)}")
    # print(f" train_mask {sum(train_mask)} {len(training_positive_samples) + len(training_negative_samples)}")
    # print(f" test_mask {sum(test_mask)} {len(test_positive_samples) + len(test_negative_samples)}")
    # print(f" val_mask {sum(val_mask)} {len(validation_positive_samples) + len(validation_negative_samples)}")
        
    return y_train, train_mask, y_test, test_mask, y_val, val_mask

def fill_labels_and_mask(positive_samples, negative_samples ,node_gene_list, y, mask):
    for gene in positive_samples:
        y[np.where(node_gene_list == gene), 0] = 1.0
        mask[np.where(node_gene_list == gene)] = 1.0 #np.where(np.array(lista) == "KIF23")[0][0]
    for gene in  negative_samples:
        mask[np.where(node_gene_list == gene)] = 1.0

def do_pca_positives_vs_negatives(host_factor_data_path, strong_host_factors):
    positive_host_factors_file = open(strong_host_factors, "r")
    positive_host_factors = []
    for line in positive_host_factors_file:
        gene = line.strip()
        positive_host_factors.append(gene)    
    
    #print(positive_host_factors)    
        
    host_factors = pd.ExcelFile(host_factor_data_path)
    host_factors_dfs = {sheet_name: host_factors.parse(sheet_name) for sheet_name in host_factors.sheet_names}
    potential_host_factors_to_remove = host_factors_dfs["host_factors"]["Gene name"].unique()
    
    json_file = open("feature_dict_absolute_value.json", "r")
    #json_file = open("feature_dict_signed_value.json", "r")
    features_dict = json.load(json_file)
    nodes_genes = list(features_dict.keys())
    negatives_feature_vectors = []
    positives_feature_vectors = []
    
    for gene in tqdm(nodes_genes):
        if gene in positive_host_factors:
            positives_feature_vectors.append(features_dict[gene])
        if gene not in positive_host_factors and gene not in potential_host_factors_to_remove:
            negatives_feature_vectors.append(features_dict[gene])  
    negatives_feature_vectors = random.sample(negatives_feature_vectors, 4*len(positives_feature_vectors))        
    compute_pca_and_plot(positives_feature_vectors, negatives_feature_vectors)
    

def feature_fold_change_pvalue_combination(log2fc, p_val, is_absolute_val):
    if p_val <= 10 ** -9:
        p_val = 10 ** -9
    if is_absolute_val:
        return np.sqrt(np.abs(log2fc * np.log10(p_val)))
    else:
        return (log2fc/np.abs(log2fc))*np.sqrt(np.abs(log2fc * np.log10(p_val)))
    

def compute_common_genes(proteome_file, stranscriptome_file):
    df_proteomics = pd.read_csv(proteome_file)
    df_transcriptomics = pd.read_csv(stranscriptome_file)
    
    common_genes = []
    for gene_name in df_proteomics["gene_name"].unique():
        if gene_name in df_transcriptomics["gene_name"].unique():
            common_genes.append(gene_name)
    return df_proteomics, df_transcriptomics, common_genes   

def extract_feature_from_row(row, time_list, feature_vector, is_used_abs_value_for_up_down_regulated, fold_change_first_part :str, p_val_first_part :str):
    for time in time_list:
        l2fc = row[ fold_change_first_part + time].values
        pval = row[ p_val_first_part + time].values
        if len(l2fc) == 0:
            l2fc.append(10 ** -9)
        if len(pval) == 0:
            pval.append(0.99)    
        feature_vector.append(feature_fold_change_pvalue_combination( l2fc[0], pval[0], is_used_abs_value_for_up_down_regulated))
    
    return feature_vector              

def proteomics_trascriptomics_features_prep(timestamp_list_transcriptomics, timestamp_list_proteomics, is_used_abs_value_for_up_down_regulated :bool):
    if os.path.isfile("feature_dict_absolute_value.json") and is_used_abs_value_for_up_down_regulated:
        return
    elif os.path.isfile("feature_dict_signed_value.json") and not is_used_abs_value_for_up_down_regulated:
        return
        
    df_proteomics, df_transcriptomics, common_genes  =  compute_common_genes("proteome_cell_lines.zip", "transcriptome.zip")
    feature_dict = {}

    for gene in tqdm(common_genes):
        feature_vector = []
        row = df_transcriptomics[df_transcriptomics["gene_name"] == gene]
        feature_vector = extract_feature_from_row(row, timestamp_list_transcriptomics, feature_vector, is_used_abs_value_for_up_down_regulated, "fold_change_log2", "p_value")
            
        row = df_proteomics[df_proteomics["gene_name"] == gene]    
        feature_vector = extract_feature_from_row(row, timestamp_list_proteomics, feature_vector, is_used_abs_value_for_up_down_regulated, "fold_change_log2", "p_value")
           
        feature_dict.update({gene : feature_vector})
    
    if is_used_abs_value_for_up_down_regulated:
        json.dump(feature_dict, open( "feature_dict_absolute_value.json", "w"))  
    elif not is_used_abs_value_for_up_down_regulated:
        json.dump(feature_dict, open( "feature_dict_signed_value.json", "w"))        
    
    

def proteomics_trascriptomics_ppi_prep():
    if os.path.isfile("df_string_transcriptomics_proteomics.zip"):
        return    
    df_proteomics, df_transcriptomics, common_genes  =  compute_common_genes("proteome_cell_lines.zip", "transcriptome.zip") 
            
    df_string_ppi = pd.read_csv("df_string_ppi_with_gene_names.zip")
    print("Removing edjes connecting genes that do not have complete transcriptomics and proteomics features")
    counter = 0
    list_of_row_indices_to_drop = []
    print(f"Counter value to {len(df_string_ppi)}")
    for index, row in df_string_ppi.iterrows():
        if (row["gene_name_1"] not in common_genes) or (row["gene_name_2"] not in common_genes):
            counter += 1
            list_of_row_indices_to_drop.append(index)
            if counter % 100000 == 0:    
                print(index)
    df_string_ppi.drop(list_of_row_indices_to_drop, axis = 0, inplace = True)            
    df_string_ppi.reset_index(drop = True, inplace = True)
    df_string_ppi.to_csv('df_string_transcriptomics_proteomics.zip', index=False, compression = dict(method='zip', archive_name='df_string_transcriptomics_proteomics.csv'))
          

def add_gene_names_to_ppi():
    if os.path.isfile("df_string_ppi_with_gene_names.zip"):
        return
        
    if not os.path.isfile("df_string_to_gen_names_map.zip"):
        print("df_string_to_gen_names_map.zip was not computed")
        
    df_string_to_gen_names_map = pd.read_csv("df_string_to_gen_names_map.zip")
    df_string_ppi = pd.read_csv("df_string_ppi.zip")
    gene_names_1 = []
    gene_names_2 = []

    for index, row in df_string_ppi.iterrows():
        names_1 = df_string_to_gen_names_map[df_string_to_gen_names_map["string_ids"] == row["protein1"]]["gene_names"]
        names_2 = df_string_to_gen_names_map[df_string_to_gen_names_map["string_ids"] == row["protein2"]]["gene_names"]
        if len(names_1) > 0:
            gene_names_1.append(names_1.values[0])
        else:
            gene_names_1.append('nan')
        if len(names_2) > 0:    
            gene_names_2.append(names_2.values[0])
        else:
            gene_names_2.append('nan')
            
    df_string_ppi["gene_name_1"] = pd.Series(gene_names_1, index = df_string_ppi.index)     
    df_string_ppi["gene_name_2"] = pd.Series(gene_names_2, index = df_string_ppi.index)  
    df_string_ppi.to_csv('df_string_ppi_with_gene_names.zip', index=False, compression = dict(method='zip',archive_name='df_string_ppi_with_gene_names.csv'))
    

def map_ensp_to_gene_name(uniprot_file_path :str):
    if os.path.isfile("df_string_to_gen_names_map.zip"):
        return
    if not os.path.isfile("df_string_ppi.zip"):
        print("df_string_ppi.zip not present")
        return
    
    uniprot_dat_file_content = [i.strip().split() for i in open(uniprot_file_path).readlines()]
    string_and_gene_names = []
    for item in uniprot_dat_file_content:
        if (item[1] == "Gene_Name") or (item[1] == "STRING"):
            string_and_gene_names.append(item)    
    
    string_names_list = []
    gene_names_list = []
    counter = 0
    print(f"Counter value to {len(string_and_gene_names)}")
    for elem in string_and_gene_names:
        if elem[1] == "STRING":
            for names in string_and_gene_names: # check if the uniprot name is the same
                if names[0] == elem[0] and names[1] == "Gene_Name":
                    string_names_list.append(elem[2])
                    gene_names_list.append(names[2])
                    break
        counter += 1
        if counter % 10000 == 0:
            print(counter)     
            
    string_to_gen_names_map = {"string_ids" : string_names_list, "gene_names" : gene_names_list}        
    df_string_to_gen_names_map = pd.DataFrame(string_to_gen_names_map)
    df_string_to_gen_names_map.to_csv('df_string_to_gen_names_map.zip', index=False, compression = dict(method='zip',archive_name='df_string_to_gen_names_map.csv'))
            

def import_and_filter_string_ppi(path_to_string :str):
    
    if os.path.isfile("df_string_ppi.zip"):
        return
    
    df_string_ppi = pd.read_csv(path_to_string , sep = " ")
    print("removing the experimental edjes")
    counter = 0
    index_list = []
    print(len(df_string_ppi))
    for index, row in df_string_ppi.iterrows():
        if row["experimental"] < 10:
            counter += 1
            index_list.append(index)
            if counter % 10000 == 0:    
                print(index) 
                 
    df_string_ppi.drop(index_list, axis = 0, inplace = True) 
    df_string_ppi.reset_index(drop = True, inplace = True) 
    
    df_string_ppi.to_csv('df_string_ppi.zip', index=False, compression = dict(method='zip',archive_name='df_string_ppi.csv'))
                

def data_preprocessing(input_data_path):
    
    if os.path.isfile("transcriptome.zip") and os.path.isfile("proteome_cell_lines.zip") and os.path.isfile("host_factors_to_remove.zip"):
        return
    
    print("Input file reading")
    data = pd.ExcelFile(input_data_path)

    dfs =  {sheet_name: data.parse(sheet_name) for sheet_name in data.sheet_names}

    transcriptome = dfs["Tx"]
    proteome_cell_lines = dfs["FP"]

    print("Host factors file reading")
    host_factors = pd.ExcelFile("../host_factors_from_publications.xlsx")

    host_factors_dfs = {sheet_name: host_factors.parse(sheet_name) for sheet_name in host_factors.sheet_names}
    host_factors_to_remove = host_factors_dfs["host_factors"]
    
    transcriptome.to_csv('transcriptome.zip', index=False, compression = dict(method='zip',archive_name='transcriptome.csv'))
    proteome_cell_lines.to_csv('proteome_cell_lines.zip', index=False, compression = dict(method='zip',archive_name='proteome_cell_lines.csv'))
    host_factors_to_remove.to_csv('host_factors_to_remove.zip', index=False, compression = dict(method='zip',archive_name='host_factors_to_remove.csv'))
    

def parse_args():
    parser = argparse.ArgumentParser(description='Data preprocessing for transcriptome and proteome from cell lines')

    parser.add_argument('--input_data_path', help='Artivir file path',
                    default="../ARTIvir_CoV_minimal_combined_dset.xlsx",
                    type=str
                    )
    parser.add_argument('--host_factor_data_path', help='Host factors file data path',
                    default="../host_factors_from_publications.xlsx",
                    type=str
                    )    
    parser.add_argument('--path_to_string', help='STRING ppi network file path',
                    default="../9606.protein.physical.links.detailed.v11.5.txt",
                    type=str
                    )    
    parser.add_argument('--uniprot_file_path', help='Uniprot download file path',
                    default="../HUMAN_9606_idmapping.dat",
                    type=str
                    )   
    parser.add_argument('--strong_host_factors', help='Host factors to consider as positive labels',
                    default="../strong_host_factors.txt",
                    type=str
                    )  
    parser.add_argument( "input_data_type", type=str, choices=["transcriptomics", "proteomics", "transcriptomics-proteomics"],
                    help="Choose which data you want to use to create the input dataset for EMOGI")
    
    args = parser.parse_args()
    return args                   


if __name__ == "__main__":
    args = parse_args()
    input_data_path = args.input_data_path
    host_factor_data_path = args.host_factor_data_path
    path_to_string = args.path_to_string
    uniprot_file_path = args.uniprot_file_path
    strong_host_factors = args.strong_host_factors
    input_data_type = args.input_data_type
    
    data_preprocessing(input_data_path)
    import_and_filter_string_ppi(path_to_string)
    map_ensp_to_gene_name(uniprot_file_path)
    add_gene_names_to_ppi()
    if input_data_type == "transcriptomics-proteomics":
        proteomics_trascriptomics_ppi_prep()
        time_stamp_list = [".SARS_CoV2@6h_vs_mock@6h", ".SARS_CoV2@12h_vs_mock@12h", ".SARS_CoV2@24h_vs_mock@24h"]
        proteomics_trascriptomics_features_prep(time_stamp_list, time_stamp_list, True)

        network, features, node_names, feat_names = create_adjacency_matrix_and_feature_matrix("df_string_transcriptomics_proteomics.zip", "feature_dict_absolute_value.json")

        y_train, train_mask, y_test, test_mask, y_val, val_mask = train_test_val_split(host_factor_data_path, strong_host_factors)
        
        create_hdf5_container(network, features, node_names, feat_names, y_train, train_mask, y_test, test_mask, y_val, val_mask, "transcriptomics_proteomics")
        #do_pca_positives_vs_negatives(host_factor_data_path, strong_host_factors)