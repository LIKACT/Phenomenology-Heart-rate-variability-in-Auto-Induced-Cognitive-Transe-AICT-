from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os.path as op
import scipy.io as sio
from tqdm import trange
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time
from scipy.stats import binom

start_time = time.time()

## params ##
matrice_name = 'mat_hrv.mat' 
FREQ = ['HRV_27']
ITERATIONS = 100

## path ##
root = op.join('/Users', 'victoroswald', 'Documents', 'code', 'Trance','result')

## load label and group ##
y= np.concatenate((np.zeros(27, dtype=int), np.ones(27, dtype=int)))
print(y.shape)
group_vector = np.arange(1, 28)
groups = np.concatenate([group_vector, group_vector])  # Concatène deux fois la séquence de 1 à 52
print(groups.shape)
# make a list out of the data split generator for random access
for freq in FREQ:
    print(freq)
    x = sio.loadmat(op.join(root, matrice_name))['mat']
    print (x.shape)
    def compute(train, test):
      
    # params
        params = {"n_estimators": [75, 100, 150, 200],
            "max_depth": [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"],
            "min_samples_leaf": [1, 2, 6, 10]}
        
    # generate a K-fold split of the training subset for cross validation during the grid search
        kfold_inner = GroupKFold(n_splits=5)
        inner_cross_validation_split = kfold_inner.split(x[train], y[train], groups[train])
    # run grid search
        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=params,
            cv=inner_cross_validation_split,
            refit=True,
            n_jobs=-1,)
     #   grid_search.fit(x[train], y[train], groups[train])
        grid_search.fit(x[train], y[train])
        return (grid_search.best_estimator_.feature_importances_,
            grid_search.best_estimator_.score(x[test] , y[test]),)
   
    
    cv = GroupKFold(n_splits=5)
    cv_split = list(cv.split(x, y, groups))
    perm = np.random.permutation(len(cv_split))

## Parallel loop on feature and frequency
    results = Parallel(n_jobs=-1)(delayed(compute)(*cv_split[perm[i % len(cv_split)]]) for i in trange(ITERATIONS))

    importances = [result[0] for result in results]
    scores = [result[1] for result in results]
    
    mean_scores = np.mean(scores)
    print(f"mean accuracy with iterations : {mean_scores}")
    nombre_total_predictions = len(y)
    nombre_estime_correctes = mean_scores * nombre_total_predictions 
    p_hasard = 0.5
    p_value_globale = 1 - binom.cdf(nombre_estime_correctes - 1, nombre_total_predictions, p_hasard)
    print(f"p-value globale du test binomial: {p_value_globale}")
    pvalue_array = np.array([p_value_globale])
    np.savetxt(root + '/p_value_'+freq+'.txt',  pvalue_array)
    
    np.savetxt(root + '/importances_HRV_study_1_' + freq + '.txt', importances)
    np.savetxt(root + '/scores_HRV_study_1_' + freq + '.txt', scores)


    end_time = time.time()  # Enregistrer l'heure de fin
    elapsed_time = end_time - start_time  # Calculer le temps écoulé
    print(f"Elapsed time : {elapsed_time:.2f} seconds")

    
