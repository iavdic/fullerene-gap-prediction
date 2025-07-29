import pandas as pd
from scripts.data_loader import load_data, featurize_dataframe
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

model = load_model('results/model.h5', compile=False)
data = load_data('data/Fullerene_Gap_Dataset.csv')
X, _ = featurize_dataframe(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Ideally reuse saved scaler stats

preds = model.predict(X_scaled)
data['predicted_gap'] = preds
os.makedirs('results', exist_ok=True)
data.to_csv('results/predictions.csv', index=False)
print(data[['smiles', 'predicted_gap']])
