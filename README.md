# NILM

## Description

Ce projet implémente la désagrégation non intrusive de la consommation électrique (NILM) pour prédire la consommation de plusieurs appareils à partir de la consommation totale.  
Deux modèles sont disponibles :

- **Energformer** : Transformer amélioré avec convolutions séparables et attention linéaire.
- **TransformerMultiOutputNILM** : Transformer standard multi-sorties.

Le projet a été testé sur un dataset industriel avec **3 channels**.

---

## 📂 Project Structure
src/
  ├── dataset.py            # Gère le chargement, le prétraitement des données et la création des datasets/dataloaders PyTorch
  ├── energformer.py        # Implémente l'architecture du modèle Energyformer
  ├── model_nilm.py         # Implémente l'architecture du modèle TransformerMultiOutputNILM
├── evaluate.py           # Script pour évaluer un modèle entraîné sur le jeu de test et calculer les métriques
├── requirements.txt      # Liste des dépendances Python nécessaires au projet
└── train.py              # Script d'entraînement du modèle (boucle d'entraînement, journalisation, sauvegarde des checkpoints)

 

## Installation

1. Cloner le dépôt :

git clone <url_du_projet>
cd <nom_du_projet>

python -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows
pip install -r requirements.txt

## Entrainement

Dans train.py, choisir le modèle :

#### Energformer

model = Energformer(...)

#### TransformerMultiOutputNILM

model = TransformerMultiOutputNILM(...)

## Evaluation

Dans evaluate.py, définir MODEL_PATH selon le modèle utilisé

## Dataset 
Pour le dataset complet, téléchargez-le ici : [Google Drive] (https://drive.google.com/drive/folders/1ceKvKgfcXC0dNSPwXUrFBPwvU_jCzuHD?usp=sharing)

