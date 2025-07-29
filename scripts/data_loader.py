import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def load_data(path):
    df = pd.read_csv(path)
    expected_cols = {'smiles', 'gap'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Missing expected columns: {expected_cols - set(df.columns)}")
    return df

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
    }

def featurize_dataframe(df):
    features = df['smiles'].apply(compute_descriptors).dropna()
    X = pd.DataFrame(features.tolist())
    y = df.loc[features.index, 'gap']
    return X, y
