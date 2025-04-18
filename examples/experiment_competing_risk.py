
# Comparsion models for competing risks
# In this script we train the different models for competing risks

#kdkoch2 modifications with reduced folds and grid and epochs
#explicitly add k=2 and so not k=5 which is default

import sys
from nfg import datasets
from experiment import *

random_seed = 0

# Open dataset
dataset = sys.argv[1] # FRAMINGHAM, SYNTHETIC_COMPETING, PBC, SEER

# Specific fold selection
fold = None
if len(sys.argv) == 3:
    fold = int(sys.argv[2])

print("Script running experiments on ", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing = True) 

# Hyperparameters
#max_epochs = 1000
#grid_search = 100
max_epochs = 200
grid_search = 5

small_layers = layers = [[25], [50], [25, 25], [50, 50]] #for DeSurv which is failing 4-6 times smaller?

layers = [[i] * (j + 1) for i in [25, 50] for j in range(4)]
layers_large = [[i] * (j + 1) for i in [25, 50] for j in range(8)]

#batch = [100, 250] if dataset != 'SEER' else [1000, 5000]  #this crashed on SEER_ds DeSurv Expt
#batch = [100, 250] if dataset != 'SEER' else [200,1000] #this crashed on SEER_ds after running 2 folds
batch = [100, 250]
# DSM
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    #'k' : [2, 3, 4, 5],
    'distribution' : ['LogNormal', 'Weibull'],
    'layers' : layers_large,
}

#kdkoch2 modifications
# Ensure the Results directory exists
results_dir = 'Results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
#End kdkoch2 modifications

DSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsm'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e)
DSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsmnc'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e == 1)

# NFG Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,
    
    'dropout': [0., 0.25, 0.5, 0.75],

    'layers_surv': layers,
    'layers' : layers,
    'act': ['Tanh'],
}
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfg'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e)
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgnc'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e == 1)
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgcs'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e, cause_specific = True)

param_grid['multihead'] = [False]
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgmono'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e)

# Desurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'embedding': [True],
    'layers_surv': layers,
    #'layers': layers,
    'layers': small_layers,  #Replaced for SEER
    'act': ['Tanh'],
}
DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_ds'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e)
DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsnc'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e == 1)

# DeepHit Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,
    
    #from layers to small_layers for SEER
    'nodes' : small_layers,
    'shared' : small_layers
}
DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dh'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e)
DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dhnc'.format(dataset), random_seed = random_seed, fold = fold,k=2).train(x, t, e == 1)