
# Importer les modules nécessaires
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os.path as op

# Charger le fichier Excel
#file_path = "/Users/victoroswald/Documents/code/Trance/result/combined_data.xlsx"
file_path = "/Users/victoroswald/Documents/code/Trance/result/combined_data_hrv.xlsx"
combined_data = pd.read_excel(file_path)
root = op.join('/Users', 'victoroswald', 'Documents', 'code', 'Trance', 'result')

print(combined_data)
# Définir les variables à filtrer
#filtered_labels = [
 #   'External thoughts', 'Innner thoughts', 'Retrospective thinking',
 #   'Prospective thinking', 'Imagined scene', 'Tactile sensation',
 #   'Visceral sensation', 'Emotion', 'Mind wandering', 'Absorption',
 #   'Altered time perception'
#]

lab = ['HF', 'LFn', 'CSI', 'Fuzzy entropy', 'LCZ complexity']
print(lab)

# Filtrer les données uniquement pour les variables sélectionnées
#grouped_data_filtered = combined_data[['time', 'group'] + filtered_labels]

# Palette de couleurs spécifique
palette = {
    'Above Median': 'blue',
    'Below Median': 'red',
    'All Participants': 'black'}

# Définir les styles de lignes : pointillé pour les groupes médian
linestyles = {
    'Above Median': '--',
    'Below Median': '--',
    'All Participants': '-'}

# Créer une figure avec des sous-graphiques (3x4 layout)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
#fig, axes = plt.subplots(4, 3, figsize=(20, 12))
axes = axes.flatten()  # Aplatir les axes pour un parcours facile

# Tracer chaque feature filtrée sur un sous-graphe
for idx, label in enumerate(lab):
    for group in ['Above Median', 'Below Median', 'All Participants']:
        sns.pointplot(
            data=combined_data[combined_data['group'] == group],
            x="time", y=label, ax=axes[idx], markers="o", linestyles=linestyles[group],
            color=palette[group], legend=False)
    axes[idx].set_title(label, fontsize=14, fontweight='bold')  # Titre en gras
    axes[idx].set_xlabel("", fontsize=12)  # Enlever le label 'Time'
    axes[idx].set_ylabel("HRV", fontsize=12)  # Ajuster la police à 12

    # Changer les labels x en gras
    axes[idx].set_xticks([0, 1])
    axes[idx].set_xticklabels(["Rest", "AICT"], fontsize=12, fontweight='bold')

    # Supprimer la boîte autour des sous-graphiques
    axes[idx].spines['top'].set_visible(False)
    axes[idx].spines['right'].set_visible(False)
    #axes[idx].spines['left'].set_visible(False)
    #axes[idx].spines['bottom'].set_visible(False)

# Supprimer les sous-graphiques inutilisés (s'il y en a)
for ax in axes[len(lab):]:
    ax.remove()

# Ajuster l'espace entre les sous-graphiques
plt.tight_layout()
plt.savefig(op.join(root, "figure_dif_inter_plot_tr_2.png"),dpi=600)
plt.show()   
    
    
    