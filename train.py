"""
Model Training and Evaluation Pipeline for Skin Disease Classification
Using UCI Dermatology Dataset and Random Forest Classifier.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DISEASE_CLASSES = {
    1: "Psoriasis",
    2: "Seborrheic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}

FEATURE_NAMES = [
    "erythema", "scaling", "definite_borders", "itching",
    "koebner_phenomenon", "polygonal_papules", "follicular_papules",
    "oral_mucosal_involvement", "knee_elbow_involvement", "scalp_involvement",
    "family_history", "melanin_incontinence", "eosinophils_infiltrate",
    "pnl_infiltrate", "fibrosis_papillary_dermis", "exocytosis",
    "acanthosis", "hyperkeratosis", "parakeratosis",
    "clubbing_rete_ridges", "elongation_rete_ridges",
    "thinning_suprapapillary_epidermis", "spongiform_pustule",
    "munro_microabcess", "focal_hypergranulosis",
    "disappearance_granular_layer", "basal_layer_damage",
    "spongiosis", "saw_tooth_appearance_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_mononuclear_infiltrate",
    "band_like_infiltrate", "age"
]

def load_data(filepath="data/dermatology.csv"):
    if not os.path.exists(filepath):
        print(f"[!] Dataset not found at {filepath}. Attempting to download...")
        from data.download_data import download_and_save
        download_and_save(filepath)

    print(f"[*] Loading dataset from {filepath}...")
    data = pd.read_csv(filepath)

    # Replace missing markers '?' with NA
    data.replace("?", pd.NA, inplace=True)
    data = data.apply(pd.to_numeric, errors="coerce")

    # Drop missing values (commonly in age column in UCI dermatology dataset)
    initial_len = len(data)
    data.dropna(inplace=True)
    print(f"[+] Loaded {len(data)} valid records (dropped {initial_len - len(data)} records with missing values).")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)

    # Set column names if header wasn't present
    if list(X.columns) != FEATURE_NAMES:
        X.columns = FEATURE_NAMES

    return X, y

def train_and_evaluate(X, y, model_path="model.pkl"):
    os.makedirs("assets", exist_ok=True)

    print("\n" + "="*50)
    print(" 1. 5-Fold Cross Validation")
    print("="*50)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
    
    print(f"Fold Accuracy Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"Mean CV Accuracy:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\n" + "="*50)
    print(" 2. Train-Test Split (80% Train / 20% Test)")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Test Set Accuracy:   {test_acc:.4f} ({test_acc*100:.1f}%)")
    print("\nClassification Report:")
    target_names = [f"{cls_id}: {DISEASE_CLASSES[cls_id]}" for cls_id in sorted(y.unique())]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix Visualization
    plt.figure(figsize=(9, 7))
    cm = confusion_matrix(y_test, y_pred)
    short_names = [DISEASE_CLASSES[i] for i in sorted(y.unique())]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names)
    plt.title("Confusion Matrix - Skin Disease Classification", fontsize=13, weight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = "assets/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to {cm_path}")

    # Top Feature Importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[FEATURE_NAMES[i] for i in indices], palette="viridis")
    plt.title("Top 15 Most Influential Diagnostic Features", fontsize=13, weight="bold")
    plt.xlabel("Random Forest Feature Importance", fontsize=11)
    plt.tight_layout()
    fi_path = "assets/feature_importance.png"
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"[+] Feature importance plot saved to {fi_path}")

    # Save trained model
    joblib.dump(rf, model_path)
    print(f"[+] Model successfully serialized to {model_path}")

if __name__ == "__main__":
    X, y = load_data()
    train_and_evaluate(X, y)
