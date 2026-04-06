import torch as pt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from .ProcrustesEncoder import ProcrustesEncoder

class procrustes_model(object):
     '''Supervised model for the parametric Procrustes transformation.
     '''
     def __init__(self, base_model_class=LinearRegression, base_model_options={}):
        self.base_model_class = base_model_class
        self.base_model_options = base_model_options
        self.models = {'mean': [], 'scale': [], 'rotation': []}
     
     def train(self, encoders, ocs):
        nocs = pt.atleast_2d(pt.Tensor(ocs))
        # ocsmin = pt.min(nocs, dim=0, keepdim=True).values
        # ocsmax = pt.max(nocs, dim=0, keepdim=True).values
        # nocs = (nocs - ocsmin) / (ocsmax-ocsmin)
        # print('nocs', nocs)

        means_train = [encoder.mean_ for encoder in encoders.values()]
        stds_train = [encoder.scale_ for encoder in encoders.values()]
        rotations_train = [encoder.rotation_.flatten() for encoder in encoders.values()]
        
        means_trains = pt.concat(means_train, dim=-1).T.unsqueeze(0)
        stds_trains = pt.Tensor(stds_train).unsqueeze(0).unsqueeze(-1)
        rotations_trains = pt.stack(rotations_train).unsqueeze(0)
        Ys = {'mean': means_trains, 'scale': stds_trains, 'rotation': rotations_trains}

        for param in self.models.keys():
            ys = Ys[param]
            for batch in range(ys.shape[-1]):
                    model_i = self.base_model_class(**self.base_model_options)
                    model_i.fit(nocs.numpy(), ys[0,..., batch].T.numpy())
                    self.models[param].append(model_i)

        return self

     def eval(self, oc):

        params_cnmc = {pm: pt.Tensor([mod.predict(oc) for mod in self.models[pm]]) 
                       for pm in self.models.keys()}
        
        mean_cnmc  = params_cnmc['mean']
        std_cnmc  = params_cnmc['scale']
        rotation_cnmc = params_cnmc['rotation']
        cnmc_encoder = ProcrustesEncoder()
        sdim = mean_cnmc.shape[0]
        cnmc_encoder._state_size = mean_cnmc.shape[0]
        cnmc_encoder.mean_ = mean_cnmc
        cnmc_encoder.scale_ = std_cnmc
        cnmc_encoder.rotation_ = rotation_cnmc.reshape((sdim, sdim))
        cnmc_encoder.trained = True 

        return cnmc_encoder
