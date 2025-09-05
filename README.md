# NILM

## Description

Ce projet implémente la désagrégation non intrusive de la consommation électrique (NILM) pour prédire la consommation de plusieurs appareils à partir de la consommation totale.  
Deux modèles sont disponibles :

- **Energformer** : Transformer amélioré avec convolutions séparables et attention linéaire.
- **TransformerMultiOutputNILM** : Transformer standard multi-sorties.

Le projet a été testé sur un dataset industriel avec **3 channels**.

---

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
