
# Comparsion models for competing risks
# In this script we train the different models for competing risks
import sys
from nfg import datasets
from experiment import *

random_seed = 0

# Open dataset
dataset = sys.argv[1] # FRAMINGHAM, SYNTHETIC_COMPETING, PBC, SEER
print("Script running experiments on ", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing = True) 

# Hyperparameters and evaluations
horizons = [0.25, 0.5, 0.75]
times = np.quantile(t[e!=0], horizons)

max_epochs = 10
grid_search = 3
layers = [[i] * (j + 1) for i in [25, 50] for j in range(4)]
layers_large = [[i] * (j + 1) for i in [25, 50] for j in range(8)]

batch = [5000]

# DeSurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-4],
    'batch': batch,
    
    'dropout': [0.],

    'layers_surv': [[50] * 3],
    'layers' : [[50] * 3],
    'act': ['Tanh'],
}
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_fg_fixed'.format(dataset), times = times, random_seed = random_seed).train(x, t, e)
for n in range(2, 100, 2):
    param_grid['n'] = [n]
    DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_ds_n={}'.format(dataset, n), times = times, random_seed = random_seed).train(x, t, e)

