#author: Roman Schulte-Sasse
# date: 20.06.2018

import pandas as pd
import numpy as np
import os, h5py, argparse, sys
from collections import Counter
from joblib import Parallel, delayed


def get_mean_betaval_for_promoters(annotation_genes, methylation_levels):
    """Extract the mean methylation level at promoters for each gene in the annotation.
    
    This function looks for all the methylation sites that fall
    inside all the promoters in the annotation genes df.
    The annotation_genes df is expected to contain the columns
    'promoter_start' and 'promoter_end'.
    
    Parameters:
    ----------
    annotation_genes:                 A dataframe containing a gene per row.
                                      It needs to have columns for promoter_start
                                      and promoter_end.
    methylation_levels:               A dataframe that contains cpg sites per row.
                                      The sites are expected to be one bp long and
                                      should have the columns: 'Start', 'End' and 
                                      'Beta_value'.
    Returns:
    Two lists with the mean beta values for each promoter and the number of cpg sites
    that support the promoter. The first list contains the mean beta values and
    the second list contains the support.
    """
    beta_vals_prom = []
    beta_vals_gene = []
    n_supports_prom = []
    n_supports_gene = []
    count = 0
    for _, row in annotation_genes.iterrows():
        # promoter
        m_sites_in_promoter = methylation_levels[methylation_levels.Start.between(row.promoter_start, row.promoter_end)]
        beta_vals_prom.append(m_sites_in_promoter.Beta_value.mean())
        n_supports_prom.append(m_sites_in_promoter.shape[0])
        # gene body
        if row.strand == '+':
            m_sites_gene = methylation_levels[methylation_levels.Start.between(row.promoter_end, row.end)]
        else:
            m_sites_gene = methylation_levels[methylation_levels.Start.between(row.start, row.promoter_start)]
        beta_vals_gene.append(m_sites_gene.Beta_value.mean())
        n_supports_gene.append(m_sites_gene.shape[0])
        count += 1

    return beta_vals_prom, beta_vals_gene, n_supports_prom, n_supports_gene


def load_annotation_gff(annotation_path, gene_list=None, meth_data=None):
    """Load a GFF annotation file from disk into a pandas DF.

    This function will read a genome annotation file (GFF3 format) from disk
    into a dataframe. All annotations that do not originate from genes will
    be removed, together with NAs and non-relevant genes (optional, only when
    'gene_list' is given).

    Parameters:
    ----------
    annotation_path:            The path to the GFF3 annotation file

    gene_list:                  Dataframe with relevant genes as columns.
                                Specifies the list of genes to keep the
                                annotation for.
                                If None, all the annotations are returned.
    meth_data:                  Specify a methylation file to use for the
                                extraction of promoter regions for each of
                                the genes. If None, a simple method is used
                                that just takes the 500 bp region around the
                                TSS.

    Returns:
    -------
    A pandas DF containing the annotations. All NAs are deleted and only
    genes are kept (type column == 'gene'). Furthermore, promoter regions
    are already extracted and gene names and IDs are present as distinct
    columns.
    If a gene_list is given, only annotations for these genes are kept
    while all other genes are removed.
    """
    a_df = pd.read_csv(annotation_path, sep='\t', skiprows=7,
                       header=None,
                       names=['chr', 'source', 'type', 'start', 'end', 'score',
                              'strand', 'phase', 'attr']
                       )
    # drop all NAs
    a_df.dropna(axis=0, inplace=True)
    # remove everything that is not a gene
    a_df = a_df[a_df.type == 'gene']
    # put gene names and ENSEMBL IDs as extra columns
    annotated_gene_ids = [i[0].split('=')[1].split('.')[0].strip() for i in a_df.attr.str.split(';')]
    annotated_gene_names = [i[3].strip().split('=')[1].strip() for i in a_df.attr.str.split(';')]
    a_df['ID'] = annotated_gene_ids
    a_df['Symbol'] = annotated_gene_names

    # remove duplicate genes (not interested in transcript level)
    a_df.drop_duplicates(subset='Symbol', inplace=True)

    # extract promoter regions for the genes
    if meth_data is None: # use simple 500bp approach
        promoter_vec = np.vectorize(get_promotor_window)
        p_starts, p_ends = promoter_vec(a_df.start, a_df.end, a_df.strand)
    elif 'promoter_start' in a_df.columns and 'promoter_end' in a_df.columns:
        p_starts = a_df.promoter_start # don't do anything
        p_ends = a_df.promoter_start
    else: # use sliding window approach from Lisa
        x = a_df.apply(promoter_window_wrapper, axis=1, m_df=meth_data)
        p_starts = np.array([i[0] for i in x])
        p_ends = np.array([i[1] for i in x])
    a_df['promoter_start'] = p_starts
    a_df['promoter_end'] = p_ends

    # remove all genes that are not in the gene list
    if not gene_list is None:
        a_df = a_df[a_df.Symbol.isin(gene_list.Symbol)]
    return a_df


def load_methylation_file(path):
    m_df = pd.read_csv(path, sep='\t')
    m_df.dropna(axis=0, inplace=True)
    return m_df


def load_relevant_genes(container_path):
    with h5py.File(container_path, 'r') as f:
        node_names = f['gene_names'][:]
    # build data frame for node names and return
    return pd.DataFrame(node_names, columns=['ID', 'Symbol'])


def promoter_window_wrapper(row, m_df):
    return get_promotor_window(row['start'], row['end'], row['strand'], m_df)


def get_promotor_window(start, end, strand, meth_data=None):
    if strand == '+':
        tss = start
    else:
        tss = end
    if not meth_data is None: # fancy method to get promoter
        scan_region = (np.max(tss-1000, 0), tss + 1000)
        return calculate_promoter_window(scan_region[0], scan_region[1], meth_data)
    else:
        return np.max(tss-250, 0), tss + 250


def calculate_promoter_window(scan_start, scan_end, meth_data, shift=50, size=200):
    best_mean_meth = None
    best_window = (None, None)
    for i in range(int(scan_start), int(scan_end-size+1), shift):
        # get mean meth for window
        m_sites_window = meth_data[meth_data.Start.between(i, i+size)]
        m_mean_window = m_sites_window.Beta_value_mean.mean()
        
        # decide on best window
        if best_mean_meth is None: # first window
            best_mean_meth = m_mean_window
            best_window = i, i+size
        else: # any of the later windows
            if abs(best_mean_meth-m_mean_window) > 0.25: # large change in mean detected
                break
            else: # make the current window the best one
                best_mean_meth = m_mean_window
                best_window = i, i+size
    return best_window


def get_filenames(meth_raw_dir):
    """Extract all valid methylation files in dir.
    """
    all_filenames = []
    for dname in os.listdir(meth_raw_dir):
        sub_dirname = os.path.join(meth_raw_dir, dname)
        if os.path.isdir(sub_dirname):
            for fname in os.listdir(sub_dirname):
                if fname.endswith('gdc_hg38.txt'):
                    all_filenames.append(os.path.join(sub_dirname, fname))
    return all_filenames


def get_mean_for_sample(annotation_df, path_name, clean=False):
    """Calculates the mean methylation DF for one sample.
    
    This function computes the mean methylation level at
    the promoter and gene body for all genes in the annotation
    dataframe using the methylation data specified in the path
    name. Note that this must be in the TCGA methylation format.
    The annotation dataframe is expected to contain the promoter
    region in the columns 'promoter_start' and 'promoter_end'.
    If the directory of the file given by 'path_name' contains
    already a file 'avg_methylation.tsv', this will be loaded
    and returned simply instead of re-extracting the methylation
    values. If clean is set to True, the extraction takes place
    in any case.
    """
    sub_dirname = os.path.dirname(path_name)
    fname = os.path.basename(path_name)
    avg_meth_result_path = os.path.join(sub_dirname,
                    'avg_methylation.tsv'
                    )
    # get the cancer type first
    cancer_type = fname.split('.')[1].split('_')[1].strip().lower()
    sample_id = fname.split('.')[-3].strip()
    if not os.path.isfile(avg_meth_result_path) or clean:
        # get mean methylation levels around promoters
        meth_df = load_methylation_file(path_name)
        beta_p, beta_g, sup_p, sup_g = get_mean_betaval_for_promoters(annotation_df, meth_df)
        # construct dataframe from results
        res = pd.DataFrame([beta_p, sup_p, beta_g, sup_g]).T
        colnames = ['mean_beta_value_promoter', 'support_promoter',
                       'mean_beta_value_gene', 'support_gene']
        cols = ['{}|{}|{}'.format(sample_id, cancer_type, i) for i in colnames]
        res.columns = cols
        res.set_index(annotation_df.Symbol, inplace=True)
        # add dataframe to overall results and write to file

        # write the average beta values (at promoters) to file
        res.to_csv(avg_meth_result_path, sep='\t')
    else:
        res = pd.read_csv(avg_meth_result_path, sep='\t')
    return res, cancer_type


def get_all_mean_betavals(annotation_df, meth_files, clean=False):
    """Calculate the mean methylation beta values for all promoters.

    Calculate the average methylation beta values for a set of promoters
    using methylation array data and genome annotation. This is done by
    extracting a promoter window of 500 bp around the promoter and then
    looking with cpg sites are located within that window.
    From there, the average is build.

    Parameters:
    ----------
    annotation_df:              The pandas dataframe containing annotations
                                for all of the genes.
                                It is expected taht the DF contains a column
                                'promoter_start' and 'promoter_end' which
                                denotes the promoter start and end sites,
                                respectively.
    meth_raw_dir:               A directory containing the downloaded data
                                from TCGA. The dir is expected to contain
                                sub-dirs with all the methylation data in them.
    clean:                      Re-compute the mean beta values for all
                                promoters even when the
                                'avg_promoter_methylation.tsv' file is already
                                present? Defaults to False.
    
    Returns:
    -------
    Three lists of lists, containing the mean methylation for all promoters,
    the mean support (#cpg sites with measurements) for all promoters and
    the cancer type of the samples. The lists have the same ordering.
    """
    count = 0
    cancer_types = []
    dfs_samples_preprocessed = []
    for path_name in meth_files:
        sub_dirname = os.path.dirname(path_name)
        fname = os.path.basename(path_name)
        avg_meth_result_path = os.path.join(sub_dirname,
                        'avg_promoter_methylation.tsv'
                        )
        # get the cancer type first
        cancer_type = fname.split('.')[1].split('_')[1].strip().lower()
        sample_id = fname.split('.')[-3].strip()
        if not os.path.isfile(avg_meth_result_path) or clean:
            # get mean methylation levels around promoters
            meth_df = load_methylation_file(path_name)
            beta_p, beta_g, sup_p, sup_g = get_mean_betaval_for_promoters(annotation_df, meth_df)
            # construct dataframe from results
            res = pd.DataFrame([beta_p, sup_p, beta_g, sup_g]).T
            colnames = ['mean_beta_value_promoter', 'support_promoter',
                           'mean_beta_value_gene', 'support_gene']
            cols = ['{}|{}|{}'.format(sample_id, cancer_type, i) for i in colnames]
            res.columns = cols
            res.set_index(annotation_df.Symbol, inplace=True)
            # add dataframe to overall results and write to file
            dfs_samples_preprocessed.append(res)
            cancer_types.append(cancer_type)

            # write the average beta values (at promoters) to file
            res.to_csv(avg_meth_result_path, sep='\t')
        else:
            res = pd.read_csv(avg_meth_result_path, sep='\t')
            dfs_samples_preprocessed.append(res)
            cancer_types.append(cancer_type)
            count += 1
        if count % 100 == 0:
            print ("Processed {} methylation samples".format(count))  

    return dfs_samples_preprocessed, cancer_types

    return dfs_samples_preprocessed, cancer_types


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Preprocess methylation data')
    parser.add_argument('-a', '--annotation',
                        help='Path to annotation file (GFF3)',
                        dest='annotation',
                        type=str
                        )
    parser.add_argument('-md', '--methylation_dir',
                        help='Path to methylation download dir',
                        dest='meth_dir',
                        type=str
                        )
    parser.add_argument('-rg', '--rel_genes',
                        help='Path to container with relevant genes',
                        dest='rel_genes',
                        default=None,
                        type=str
                        )
    parser.add_argument('-ps', '--promoter_sliding',
                        help='Use a sliding window approach to extract promoter regions?',
                        dest='p_sliding_window',
                        default=False,
                        type=bool
                        )
    parser.add_argument('-mm', '--mean_methylation',
                        help='Path to file containing averaged methylation across all samples (only relevant for sliding window approach for promoter extraction)',
                        dest='mm_path',
                        type=str
                        )
    parser.add_argument('-c', '--clean',
                        help='Re-compute when there are already avg files?',
                        dest='clean',
                        default=False,
                        type=bool
                        )
    parser.add_argument('-o', '--output',
                        help='Path to output Dataframe (huge sample-wise matrix)',
                        dest='output',
                        type=str
                        )
    parser.add_argument('-n', '--cores',
                        help='Number of cores to use',
                        dest='n_jobs',
                        default=1,
                        type=int
                        )
    args = parser.parse_args()

    # load data and compute promoter windows if required
    if not args.rel_genes is None:
        relevant_genes = load_relevant_genes(args.rel_genes)
    else:
        relevant_genes = None
    if args.annotation.endswith('.tsv'):
        annotation_df = pd.read_csv(args.annotation, sep='\t')
    elif args.annotation.endswith('.gff3'):
        if args.p_sliding_window and os.path.isfile(args.mm_path):
            print ("Calculating Promoters with sliding window (Takes Time))")
            avg_meth = load_methylation_file(args.mm_path)
            annotation_df = load_annotation_gff(args.annotation,
                                                gene_list=relevant_genes,
                                                meth_data=avg_meth)
            # write promoter annotation back to disk
            annotation_df.to_csv(args.annotation + '.tsv', sep='\t')
            print ("Wrote annotations with promoters to {}".format(args.annotation + '.tsv'))
        else:
            annotation_df = load_annotation_gff(args.annotation, gene_list=relevant_genes)
    else:
        print ("Unknown Annotation file format. Only tsv and gff3 are supported")
        sys.exit(-1)
    print ("Loaded Annotation file with {} genes".format(annotation_df.shape[0]))

    # get the preprocessed samples (this is time-consuming)
    all_files = get_filenames(args.meth_dir)
    results = Parallel(n_jobs=args.n_jobs)(delayed(get_mean_for_sample)(annotation_df,
                                                                        f,
                                                                        args.clean) for f in all_files)
    all_samples_preprocessed = [i[0] for i in results]
    cancer_types = [i[1] for i in results]

    # join samples to form one df with all samples as columns
    total_df = pd.concat(all_samples_preprocessed, axis=1)
    total_df.to_csv(args.output, sep='\t')
    
    # some information on how many samples we have
    for key, value in Counter(cancer_types).items():
        print ("Found {} samples for Cancer {}".format(value, key))

    print ("Finished. Written mean methylation matrix for {} samples to disk".format(len(all_samples_preprocessed)))
    