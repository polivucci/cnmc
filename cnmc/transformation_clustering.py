from torch import cat
from sklearn.cluster import KMeans
from pandas import read_csv, DataFrame
from copy import deepcopy

def train_encoders(data_train, encoders):
    encoders_train = []
    for data, encoder in zip(data_train, encoders):
        encoder.train(data)
        encoders_train.append(encoder)
    return encoders_train

def collect_parameters(encoders):
    stds_train = []
    means_train = []
    rotations_train = []
    for encoder in encoders:
        stds_train.append(encoder.scale_)
        means_train.append(encoder.mean_)
        rotations_train.append(encoder.rotation_.flatten())
    return stds_train, means_train, rotations_train

def train_clusters(clustering, data_train, output_csv='clusters.csv'):

    # cluster all data:
    all_data = cat(data_train, dim=1)
    
    # compute clusters:
    clustering_all = clustering['algorithm'].fit(all_data.T.numpy()) # batch dimension comes first in Scikit-Learn

    # save to disc
    # makedirs(output_csv, exist_ok=True)
    DataFrame(clustering_all.cluster_centers_).to_csv(output_csv)

    return clustering

def read_clusters_from_file(cluster_csv, **kwargs_kmeans):
    # read in clusters:
    clusts_all = read_csv(cluster_csv, header=0, index_col=0)
    kwargs_kmeans = {**kwargs_kmeans, 'init': clusts_all.to_numpy()}
    clustering_all = KMeans(**kwargs_kmeans).fit(clusts_all.to_numpy())  
    return clustering_all

def train_croms(roms, data_train, encoders, clustering_all, encode=False):
    # Setup and fit n CNMs with pre-computed cluster centres and encoding.
    
    for rom, data, encoder in zip(roms, data_train, encoders):
        # print('Fitting CNM on training parameter '+str(PARAMETERS[m]))
        if encode: data = encoder.encode(data)
        rom.encoder = encoder
        rom.cluster_config = deepcopy(clustering_all)
        rom.train(reduced_state=data)

        nnz = 0
        for value in rom.Q.values():
            nnz += value.shape[0]
        print('nnz =', nnz)

    return roms