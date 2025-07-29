# Fullerene HOMO-LUMO Gap Prediction

This project provides a machine learning workflow to predict HOMO-LUMO energy gaps of fullerene molecules (e.g., C60, C70) from SMILES using molecular descriptors generated via RDKit. Initial data set taken from S. Nagasawa, E. Al-Naamani, A. Saeki, Computer-aided screening of conjugated polymers for organic solar cell: classification by random forest. J. Phys. Chem. Lett. 9, 2639–2646 (2018).

Note: Results are preliminary and will be refined in future iterations through model improvements and dataset expansion.

Inspired by [RDKit-descriptors-for-HOMO-LUMO-energy-gap-prediction](https://github.com/gashawmg/RDKit-descriptors-for-HOMO-LUMO-energy-gap-prediction).

## Features

- Data loading and descriptor generation with RDKit
- Neural network model using TensorFlow/Keras
- Preprocessing pipeline with scaling
- Prediction and visualization tools
- Example dataset with common fullerenes

## Installation

```bash
git clone https://github.com/iavdic/fullerene-gap-prediction.git
cd fullerene-gap-prediction
pip install -r requirements.txt
