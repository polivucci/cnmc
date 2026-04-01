from torch import cat
from sklearn.cluster import KMeans
from flowtorch.rom import CNM
from pandas import read_csv, DataFrame


def train_encoders(data_train, encoder_class, **encoder_kwargs):

    encoders_train = []
    for data in data_train:
        encoder = encoder_class()
        encoder.train(data, **encoder_kwargs)
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

def train_clusters(data_train, encoders, output_csv='clusters.csv', **kwargs_kmeans):

    data_encoded = []
    for data, encoder in zip(data_train, encoders):
        data_encoded.append(encoder.encode(data))

    # cluster all data:
    all_data = cat(data_encoded, dim=1)
    print(all_data.shape)

    # compute clusters:
    kwargs_kmeans={**kwargs_kmeans, 'init': 'k-means++'}
    clustering_all = KMeans(**kwargs_kmeans).fit(all_data.T.numpy()) # batch dimension comes first in Scikit-Learn

    # save to disc
    # makedirs(output_csv, exist_ok=True)
    DataFrame(clustering_all.cluster_centers_).to_csv(output_csv)

    return clustering_all

def read_clusters_from_file(cluster_csv, **kwargs_kmeans):
    # read in clusters:
    clusts_all = read_csv(cluster_csv, header=0, index_col=0)
    kwargs_kmeans = {**kwargs_kmeans, 'init': clusts_all.to_numpy()}
    clustering_all = KMeans(**kwargs_kmeans).fit(clusts_all.to_numpy())  
    print(clustering_all.cluster_centers_.shape)
    return clustering_all

def train_cnms(data_train, encoders, clustering_all, **kwargs_cnm):
    # Setup and fit n CNMs with pre-computed cluster centres and encoding.

    cnmlist_train = []
    for data, encoder in zip(data_train, encoders):
        # print('Fitting CNM on training parameter '+str(PARAMETERS[m]))
        cnm = CNM(reduced_state=data,
                  encoder=encoder, 
                  cluster_config={'algorithm': clustering_all,}, # precomputed clustering
                  **kwargs_cnm
                  )
        cnmlist_train.append(cnm)

        nnz = 0
        for value in cnm.Q.values():
            nnz += value.shape[0]
        print('nnz =', nnz)

        return cnmlist_train