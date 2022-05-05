__version__ = "0.1.0"
import pickle
import pandas as pd
import os
import joblib

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


DF = pd.read_pickle(os.path.join(THIS_DIR, "preprocessed_data_w_projected"))

continuos_features = [
    "Size PP [nm]",
    "Concentration [mg/mL]",
    "Energy Density [J/mL]",
    "abs_zeta",
    "iep_stability",
    "BET [m2/g]",
    "Isoelectric Point",
    "Zeta Pot[mV]",
]
cat_features = ["Coating"]
target = ["log_z_av"]

all_features = continuos_features + cat_features

FEATURES = continuos_features + cat_features

DF = DF.drop_duplicates(subset=FEATURES)

ESTIMATORS = joblib.load(
    os.path.join(THIS_DIR, "20210501_ensemble_wo_type_w_coating_w_pp.joblib")
)
isomap_REDUCER = joblib.load(os.path.join(THIS_DIR, "isomap_reducer.joblib"))
PCA_REDUCER = joblib.load(os.path.join(THIS_DIR, "pca_reducer.joblib"))
SCALER = ESTIMATORS[0].steps[0][1]
