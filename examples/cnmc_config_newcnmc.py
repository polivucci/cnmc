# Load data
DATAPATH = '/home/paolo/Desktop/semaan/semaan-project/cnm/hands-on/flowtorch/lorenz/lorenz/data/'
FILENAMES = [
            'Lorenz_b30.000.h5',
            'Lorenz_b40.000.h5',
            'Lorenz_b50.000.h5',
            'Lorenz_b60.000.h5',
            'Lorenz_b70.000.h5',
             ]

ti_train, tf_train = -400_000, -1
ti_test, tf_test = -800_000, -400_000

# fully working case (Example 7)
PARAMETERS = [(30.0,), (40.0,), (50.0,), (60.0,), (70.0,)]

casepath = './numerical_experiments/lorenz/'

ocs_train = [(30.0,), (40.0,), (60.0,), (70.0,)]
ocs_test = [(50.0,),]

# output config
# casepath = './numerical_experiments/ex2/bigcluster_fixed_splines/'
# outpath=casepath+'K{:d}_L{:d}/'.format(K,L)

# cnm/cnmc set up
K=14; L=1
n_init = 10
max_iter = 1_000
centroid_model='PiecewiseLinear'
# transition_model='Polynomial'
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
transition_model = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=3), Lasso(1e-2))

spline_order = 1

# from MaxStdEncoder import MaxStdEncoder
# encoder_class = MaxStdEncoder
from cnmc.encoders.ProcrustesEncoder import ProcrustesEncoder
encoder_class = ProcrustesEncoder

init = True
compute_clusters = False        # compute clusters (big and not)
compute_ocs_clusters = True    # compute own clusters for each oc
bigcluster_test = False         # use bigclusters for test CNM (instead of own clusters) (prolly needed for QT model visuals) 

# manually reorder tensor labels for K=14
reorder_labels = True
# label_order = range(14)
# label_order = [9, 14, 7, 11, 2, 13, 5, 4, 10, 6, 3, 1, 8, 12]  
label_order = [13, 2, 11, 7, 14, 9, 5, 4, 10, 6, 3, 1, 8, 12]