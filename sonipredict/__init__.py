__version__ = "0.1.0"
import pickle
import pandas as pd
import os
import joblib

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


DF = pd.read_pickle(os.path.join(THIS_DIR, "data_for_app"))

META_ANALYSIS_MODEL = {
    "continuos_features": [
        "Size PP [nm]",
        "Concentration [mg/mL]",
        "Energy Density [J/mL]",
        "Isoelectric Point",
        "Zeta Pot[mV]",
        "Volume [mL]",
        "Total Energy [J]",
    ],
    "cat_features": ["Coating", "Particle"],
}

target = ["log_z_av"]

FEATURES = (
    META_ANALYSIS_MODEL["continuos_features"] + META_ANALYSIS_MODEL["cat_features"]
)

DF = DF.drop_duplicates(subset=FEATURES)

ESTIMATORS = joblib.load(os.path.join(THIS_DIR, "model"))
isomap_REDUCER = joblib.load(os.path.join(THIS_DIR, "isomap"))
PCA_REDUCER = joblib.load(os.path.join(THIS_DIR, "pca"))
SCALER = ESTIMATORS[0].steps[0][1]
