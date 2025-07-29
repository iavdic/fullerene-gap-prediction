from scripts.data_loader import load_data, featurize_dataframe
from scripts.model import build_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

data = load_data('data/Fullerene_Gap_Dataset.csv')
X, y = featurize_dataframe(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = build_model(X.shape[1])
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")

os.makedirs('results', exist_ok=True)
model.save('results/model.h5')
pd.DataFrame(scaler.mean_).to_csv('results/scaler_mean.csv')
