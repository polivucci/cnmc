from torch import cat
from sklearn.cluster import KMeans
from pandas import read_csv, DataFrame
from copy import deepcopy

def train_encoders(data_train, encoders):
    encoders_train = {}
    for (oc, data), encoder in zip(data_train.items(), encoders):
        encoder.train(data)
        encoders_train[oc] = encoder
    return encoders_train

def collect_parameters(encoders):
    stds_train = {}
    means_train = {}
    rotations_train = {}
    for oc, encoder in encoders.items():
        stds_train[oc] = encoder.scale_
        means_train[oc] = encoder.mean_
        rotations_train[oc] = encoder.rotation_.flatten()
    return stds_train, means_train, rotations_train

def train_clusters(clustering, data_train, output_csv='clusters.csv'):

    # cluster all data:
    all_data = cat(tuple(data_train.values()), dim=1)
    
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

    for rom, oc in zip(roms, data_train.keys()):
        # print('Fitting CNM on training parameter '+str(PARAMETERS[m]))
        data = data_train[oc]
        if encode: data = encoders[oc].encode(data_train[oc])
        rom.encoder = encoders[oc]
        rom.cluster_config = deepcopy(clustering_all)
        rom.train(reduced_state=data)

        nnz = 0
        for value in rom.Q.values():
            nnz += value.shape[0]
        print('nnz =', nnz)

    return roms