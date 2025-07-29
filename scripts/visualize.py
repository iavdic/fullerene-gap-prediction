import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results/predictions.csv')

plt.figure(figsize=(6, 6))
plt.scatter(df['gap'], df['predicted_gap'], alpha=0.7)
plt.xlabel('True HOMO-LUMO Gap (eV)')
plt.ylabel('Predicted HOMO-LUMO Gap (eV)')
plt.title('Prediction Performance')
plt.plot([df['gap'].min(), df['gap'].max()],
         [df['gap'].min(), df['gap'].max()], '--r')
plt.grid(True)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/performance_plot.png')
plt.show()
