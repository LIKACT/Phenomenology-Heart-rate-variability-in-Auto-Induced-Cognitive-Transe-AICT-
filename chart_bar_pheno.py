import os.path as op
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
root = op.join('/Users', 'victoroswald', 'Documents', 'code', 'Trance', 'result')
## Charger les données 1 ##
#matrice_name = 'mat_hrv.mat'
#matrice_NP = 'mat_dif_NP.mat'
#x1 = sio.loadmat(op.join(root, matrice_name))['new_zscore'][:27,:]
#x2 = sio.loadmat(op.join(root, matrice_name))['new_zscore'][27:,:]
matrice_name = 'mat_NP_visu.mat'
x1 = sio.loadmat(op.join(root, matrice_name))['new_mat_1']
x2 = sio.loadmat(op.join(root, matrice_name))['new_mat_2']

#labels = ['ECG rate', 'RMSSD', 'LF', 'HF', 'LF/HF', 'LFn', 'HFn', 'SD1/SD2', 'CVI', 'CSI','Fuzzy Entropy', 'LCZ Complexity', #'Baevsky Index']

labels = ['External thoughts', 'Innner thoughts', 'Retrospective thinking', 'Prospective thinking', 'Focus', 'Imagined scene', 'Inner speech', 'Ambient noise', 'Tactile sensation', 'Visceral sensation', 'Tasted things', 'Taste imagination', 'Emotion', 'Mind-wandering', 'Smell imagination', 'Visualisations', 'Absorption', 'Dissociation', 'Altered time perception']

#labels = ['External thoughts', 'Innner thoughts', 'Retrospective thinking', 'Prospective thinking', 'Focus', 'Imagined scene', 'Inner speech', 'Ambient noise', 'Tactile sensation', 'Visceral sensation', 'Emotion', 'Mind-wandering', 'Visualisations', 'Absorption', 'Dissociation', 'Altered time perception']


# Convertir les matrices en DataFrames
x1 = pd.DataFrame(x1, columns=labels)
x2 = pd.DataFrame(x2, columns=labels)

# Ajout d'une colonne pour indiquer le groupe et SubjectID
x1['Group'] = 'Rest' 
x2['Group'] = 'AICT'
x1['SubjectID'] = range(1, len(x1) + 1)
x2['SubjectID'] = range(1, len(x2) + 1)

# Fusion des DataFrames
data_combined = pd.concat([x1, x2], ignore_index=True)
  
# Transformation des données au format long
data_long = pd.melt(data_combined, id_vars=['Group', 'SubjectID'], var_name='Feature', value_name='Value')

# Filtrer les variables non souhaitées
data_long = data_long[~data_long['Feature'].isin(['Tasted things', 'Taste imagination', 'Smell imagination'])]

# Calculer les moyennes et écarts-types pour chaque caractéristique et chaque groupe
stats = data_long.groupby(['Feature', 'Group']).agg({'Value': ['mean', 'std']}).reset_index()
stats.columns = ['Feature', 'Group', 'Mean', 'Std']

# Ordre souhaité des caractéristiques
#order_labels = ['Baevsky Index','ECG rate','CSI','LF/HF','LFn','LF','SD1/SD2','CVI','RMSSD','HFn','HF','LCZ Complexity','Fuzzy Entropy']
order_labels = ['External thoughts','Visceral sensation','Imagined scene','Tactile sensation','Emotion','Dissociation','Absorption', 'Visualisations','Innner thoughts','Altered time perception','Inner speech','Retrospective thinking','Prospective thinking','Focus', 'Ambient noise','Mind-wandering']

# Filtrer pour obtenir les valeurs moyennes pour chaque groupe dans l'ordre souhaité
stats = stats.set_index('Feature').loc[order_labels].reset_index()

means_rest = stats[stats['Group'] == 'Rest']['Mean'].values
means_sict = stats[stats['Group'] == 'AICT']['Mean'].values

# Ajouter la première valeur à la fin pour boucler le graphique
features = stats['Feature'].unique()
means_rest = np.concatenate((means_rest, [means_rest[0]]))
means_sict = np.concatenate((means_sict, [means_sict[0]]))
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

# Création du graphique en toile d'araignée
fig, ax = plt.subplots(figsize=(10.5, 10), subplot_kw=dict(polar=True))

ax.fill(angles, means_rest, color='lightcoral', alpha=0.5)
ax.plot(angles, means_rest, color='lightcoral', linewidth=2, linestyle='solid', label='Rest')
ax.fill(angles, means_sict, color='turquoise', alpha=0.5)
ax.plot(angles, means_sict, color='turquoise', linewidth=2, linestyle='solid', label='AICT')

# Ajouter les labels en gras
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=12, fontweight='bold', zorder=3)

for label, angle in zip(ax.get_xticklabels(), angles):
    label.set_horizontalalignment('center')
    label.set_position((label.get_position()[0], label.get_position()[1] - 0.05))  # Ajustez ici pour déplacer plus à l'extérieur


# Réglage des valeurs de l'échelle
#ax.set_ylim(0, max(max(means_rest), max(means_sict)) + 0.1)  # Ajustez les limites si nécessaire
#ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Exemple pour un espacement régulier des graduations

plt.title("Self-Reported Phenomenology during Rest and AICT", size=20, color='black', y=1.1)
#plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
#plt.legend(loc='upper center', bbox_to_anchor=(1, 0.98), ncol=2, fontsize=16)
plt.savefig(op.join(root, "figure_33.png"),dpi=600)
plt.show()

