"""
Dataset Downloader and Preparation Script for UCI Dermatology Dataset.
Downloads the benchmark dataset from the UCI Machine Learning Repository
and prepares it as a clean CSV with readable feature names.
"""

import os
import urllib.request
import zipfile
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/33/dermatology.zip"
RAW_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"

FEATURE_NAMES = [
    "erythema",
    "scaling",
    "definite_borders",
    "itching",
    "koebner_phenomenon",
    "polygonal_papules",
    "follicular_papules",
    "oral_mucosal_involvement",
    "knee_and_elbow_involvement",
    "scalp_involvement",
    "family_history",
    "melanin_incontinence",
    "eosinophils_in_infiltrate",
    "pnl_infiltrate",
    "fibrosis_of_papillary_dermis",
    "exocytosis",
    "acanthosis",
    "hyperkeratosis",
    "parakeratosis",
    "clubbing_of_rete_ridges",
    "elongation_of_rete_ridges",
    "thinning_of_suprapapillary_epidermis",
    "spongiform_pustule",
    "munro_microabcess",
    "focal_hypergranulosis",
    "disappearance_of_granular_layer",
    "vacuolisation_and_damage_of_basal_layer",
    "spongiosis",
    "saw_tooth_appearance_of_retes",
    "follicular_horn_plug",
    "perifollicular_parakeratosis",
    "inflammatory_mononuclear_infiltrate",
    "band_like_infiltrate",
    "age",
    "class"
]

def download_and_save(output_csv="data/dermatology.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print("[*] Downloading UCI Dermatology dataset...")
    try:
        # Try fetching direct data file
        df = pd.read_csv(RAW_DATA_URL, header=None, names=FEATURE_NAMES, na_values="?")
        df.to_csv(output_csv, index=False)
        print(f"[+] Successfully saved clean dataset to {output_csv} ({len(df)} records)")
        return
    except Exception as e:
        print(f"[!] Direct link failed ({e}), attempting zip download...")

    try:
        zip_path = "data/dermatology.zip"
        urllib.request.urlretrieve(UCI_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw")
        
        # Read the extracted data
        data_file = "data/raw/dermatology.data"
        if os.path.exists(data_file):
            df = pd.read_csv(data_file, header=None, names=FEATURE_NAMES, na_values="?")
            df.to_csv(output_csv, index=False)
            print(f"[+] Extracted and saved clean dataset to {output_csv}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print(f"[!] Error fetching dataset: {e}")
        print("Please place 'dermatology.csv' manually in the 'data/' folder.")

if __name__ == "__main__":
    download_and_save()
