# Load data
DATAPATH = './data/'
casepath = './output/'
FILENAMES = [
            'Lorenz_b30.000.h5',
            'Lorenz_b40.000.h5',
            'Lorenz_b50.000.h5',
            'Lorenz_b60.000.h5',
            'Lorenz_b70.000.h5',
             ]
PARAMETERS = [(30.0,), (40.0,), (50.0,), (60.0,), (70.0,)]

# train/test trajectory split
ti_train = -400_000
ti_test = -800_000

# train/test OC
ocs_train = [(30.0,), (40.0,), (60.0,), (70.0,)]
ocs_test = [(50.0,),]

# cnm/cnmc set up
K=14    # no. clusters
L=1     # no. past delays (markov order)
n_init = 10
kmeans_max_iter = 1_000
cnm_spline_order = 1

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Lasso
transition_model = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=3), Lasso(1e-2))

from cnmc.encoders.ProcrustesEncoder import ProcrustesEncoder
encoder_class = ProcrustesEncoder

compute_clusters = False        # compute clusters (big and not)
compute_ocs_clusters = True     # compute own clusters for each oc