# utils/processing_rf.py

import os
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

AMINO_ACID_SMILES = {
    'A': 'CC(C(=O)O)N', 'C': 'C(C(=O)O)N', 'D': 'CC(C(=O)O)N',
    'E': 'CCC(C(=O)O)N', 'F': 'CC1=CC=CC=C1C(=O)O', 'G': 'C(C(=O)O)N',
    'H': 'CC1=CNC=N1', 'I': 'CC(C)CC(=O)O', 'K': 'CCCC(C(=O)O)N',
    'L': 'CC(C)C(=O)O', 'M': 'CCSC(=O)O', 'N': 'CC(C(=O)O)N',
    'P': 'CC1CC1C(=O)O', 'Q': 'CCC(C(=O)O)N', 'R': 'CCCNC(=N)N',
    'S': 'COC(=O)O', 'T': 'CC(O)C(=O)O', 'V': 'CC(C)C(=O)O',
    'W': 'CC1=CNC2=CC=CC=C12', 'Y': 'CC1=CC=C(O)C=C1'
}

def sequence_to_smiles(seq):
    return '.'.join([AMINO_ACID_SMILES.get(aa, '') for aa in seq if aa in AMINO_ACID_SMILES])

def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return np.array(fp)

def preprocess_and_split(data_path="data/train.xlsx", test_size=0.2, random_state=42):
    df = pd.read_excel(data_path, converters={"Sequence": str})
    df["SMILES"] = df["Sequence"].apply(sequence_to_smiles)
    df["Fingerprint"] = df["SMILES"].apply(smiles_to_fingerprint)
    X = np.stack(df["Fingerprint"].values)
    y = df["Label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/rf_scaler.pkl")

    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

def train_and_save_rf(X_train, y_train, model_path="models/random_forest_model.pkl"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)