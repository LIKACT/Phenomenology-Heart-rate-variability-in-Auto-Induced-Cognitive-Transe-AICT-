import pandas as pd
import os.path as op
import scipy.io as sio
import statsmodels.formula.api as smf
import os
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Charger les données depuis le fichier .mat
root = op.join('/Users', 'victoroswald', 'Documents', 'code', 'Trance', 'result')
#matrice_name = 'mat_dif_NP_2.mat'
matrice_name = 'mat_hrv.mat'

# Charger les matrices x1 (Temps 1) et x2 (Temps 2)
data = sio.loadmat(op.join(root, matrice_name))
#x1 = data['rs']  # Mesures à Temps 1
#x2 = data['tr']  # Mesures à Temps 2

x1 = data['mat'][:27,:]
x2 = data['mat'][27:,:]

# Labels des variables
#labels = ['External_thoughts', 'Innner_thoughts', 'Retrospective_thinking', 'Prospective_thinking', 'Focus', #'Imagined_scene', 'Inner_speech', 'Ambient_noise', 'Tactile_sensation', 'Visceral_sensation', 'Emotion', #'Mind_Wandering', 'Visualisations', 'Absorption', 'Dissociation', 'Altered_time_perception']

labels = ['ECG_rate', 'RMSSD', 'LF', 'HF', 'LF_HF', 'LFn', 'HFn', 'SD1_SD2', 'CVI', 'CSI','Fuzzy_Entropy', 'LCZ_Complexity', 'Baevsky_Index']

# Convertir les matrices en DataFrames avec les nouveaux noms de colonnes
Y_T1 = pd.DataFrame(x1, columns=labels)  # Temps 1
Y_T2 = pd.DataFrame(x2, columns=labels)  # Temps 2

# Ajouter un identifiant de sujet
Y_T1['subject'] = range(1, len(Y_T1) + 1)
Y_T2['subject'] = range(1, len(Y_T2) + 1)

# Ajouter une colonne 'time' pour identifier Temps 1 et Temps 2
Y_T1['time'] = 1  # Temps 1
Y_T2['time'] = 2  # Temps 2

# Combiner les deux DataFrames pour créer un format long
combined_data = pd.concat([Y_T1, Y_T2])

# Normaliser les données en utilisant des scores Z
for label in labels:
    combined_data[label] = zscore(combined_data[label])

output_excel_path = op.join('/Users', 'victoroswald', 'Documents', 'code', 'Trance', 'result', 'combined_data.xlsx')

# Save the DataFrame to Excel
combined_data.to_excel(output_excel_path, index=False)

# Créer un répertoire pour sauvegarder les graphiques et résultats
output_dir = op.join(root, 'interaction_plots')
os.makedirs(output_dir, exist_ok=True)

# Fonction pour ajuster le modèle et calculer la corrélation intercept-pente
def calculate_correlation(data, label):
    try:
        model = smf.mixedlm(f"{label} ~ time", data, groups=data["subject"])
        result = model.fit()
        
        intercept = result.params['Intercept']
        slope = result.params['time']
        
        cov_matrix = result.cov_params()
        cov_int_slope = cov_matrix.loc['Intercept', 'time']
        sd_intercept = np.sqrt(cov_matrix.loc['Intercept', 'Intercept'])
        sd_slope = np.sqrt(cov_matrix.loc['time', 'time'])
        
        correlation = cov_int_slope / (sd_intercept * sd_slope)
        return correlation, result
    except Exception as e:
        # Retourner NaN si le modèle ne converge pas
        print(f"Model failed to converge for {label}: {e}")
        return np.nan, None

# Boucle sur chaque variable pour ajuster le modèle et générer les résultats
for label in labels:
    print(label)
    
    # Calculer la corrélation originale et ajuster le modèle
    observed_correlation, result = calculate_correlation(combined_data, label)
    
    if np.isnan(observed_correlation):
        print(f"Skipping {label} due to model fitting issues.")
        continue  # Passer à la variable suivante si le modèle n'a pas convergé
    
    # Liste pour stocker les corrélations bootstrap
    bootstrap_correlations = []
    n_iterations = 400  # Nombre d'itérations pour le bootstrap
    
    # Bootstrap pour créer des échantillons et recalculer les corrélations
    for _ in range(n_iterations):
        # Créer un échantillon bootstrap avec remise
        bootstrap_data = combined_data.sample(n=len(combined_data), replace=True)
        
        # Recalculer la corrélation pour l'échantillon bootstrap
        bootstrap_correlation, _ = calculate_correlation(bootstrap_data, label)
        
        # Vérifier si le modèle a convergé pour cet échantillon bootstrap
        if not np.isnan(bootstrap_correlation):
            bootstrap_correlations.append(bootstrap_correlation)
    
    # Si aucune corrélation bootstrap n'a été calculée, passer à la variable suivante
    if len(bootstrap_correlations) == 0:
        print(f"No valid bootstrap samples for {label}. Skipping.")
        continue
    
    # Calculer les intervalles de confiance (IC 95%) pour la corrélation
    lower_bound = np.percentile(bootstrap_correlations, 2.5)
    upper_bound = np.percentile(bootstrap_correlations, 97.5)
    
    # Sauvegarder les résultats du modèle principal, la corrélation observée, et les intervalles de confiance dans un fichier texte
    with open(os.path.join(output_dir, f"{label}_model_results.txt"), 'w') as f:
        # Sauvegarder les résultats du modèle principal
        f.write(result.summary().as_text())
        f.write("\n")
        
        # Sauvegarder la corrélation observée et les intervalles de confiance
        f.write(f"Observed correlation between Intercept and Slope (time): {observed_correlation:.4f}\n")
        f.write(f"95% Confidence Interval based on bootstrap: [{lower_bound:.4f}, {upper_bound:.4f}]\n")
