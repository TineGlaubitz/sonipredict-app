import plotly.express as px
import pandas as pd
import numpy as np
from . import FEATURES, DF, SCALER, ESTIMATORS
import os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
DF_SUBSET = DF[FEATURES + ["pc_1", "pc_2", "isomap_1", "isomap_2"]]


def scale(data):
    return SCALER.transform(data)


def predict(X):
    predictions = []
    for estimator in ESTIMATORS:
        predictions.append(np.exp(estimator.predict(X)))

    return {
        "predictions": predictions,
        "mean_prediction": np.mean(predictions, axis=0)[0],
        "std_prediction": np.std(predictions, axis=0)[0],
    }


def make_prediction(data):
    prediction = predict(data)
    return f"The model predicts a particle size of {prediction['mean_prediction']:.0f} nm with an uncertainty of {prediction['std_prediction']:.0f} nm."


def get_isomap_frame(data):
    data_scaled = scale(data)
    isomapped = isomap_REDUCER.transform(data_scaled)

    f_dict = dict(zip(FEATURES, data))
    f_dict["isomap_1"] = isomapped[0, 0]
    f_dict["isomap_2"] = isomapped[0, 1]

    return pd.DataFrame([f_dict])


def get_pca_frame(data):
    data_scaled = scale(data)
    pcaed = PCA_REDUCER.transform(data_scaled)

    f_dict = dict(zip(FEATURES, data))
    f_dict["pc_1"] = pcaed[0, 0]
    f_dict["pc_2"] = pcaed[0, 1]

    return pd.DataFrame([f_dict])


def get_isomap_plot(new_points=None):
    fig = px.scatter(
        DF_SUBSET,
        x="isomap_1",
        y="isomap_2",
        color_discrete_sequence=["gray"],
        size_max=55,
        hover_data=FEATURES,
        labels={
            "isomap_1": "first isomap component",
            "isomap_2": "second isomap component",
        },
    )

    if new_points is not None:
        isomap_df = get_isomap_frame(new_points)
        fig.add_trace(
            px.scatter(
                isomap_df,
                x="isomap_1",
                y="isomap_2",
                color_discrete_sequence=["red"],
                size_max=55,
                hover_data=FEATURES,
                labels={
                    "isomap_1": "first isomap component",
                    "isomap_2": "second isomap component",
                },
            ).data[0]
        )
        fig.update_layout(transition_duration=500)

    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )

    return fig


def get_pca_plot(new_points=None):
    fig = px.scatter(
        DF_SUBSET,
        x="pc_1",
        y="pc_2",
        color_discrete_sequence=["gray"],
        size_max=55,
        hover_data=FEATURES,
        labels={
            "pc_1": "first principal component",
            "pc_2": "second principal component",
        },
    )

    if new_points is not None:
        pca_df = get_pca_frame(new_points)
        fig.add_trace(
            px.scatter(
                pca_df,
                x="pc_1",
                y="pc_2",
                color_discrete_sequence=["red"],
                size_max=55,
                hover_data=FEATURES,
                labels={
                    "pc_1": "first principal component",
                    "pc_2": "second principal component",
                },
            ).data[0]
        )
        fig.update_layout(transition_duration=500)

    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )

    return fig


def run_energy_sweep(params, e_range, n_points):
    ed = float(params["Energy Density [J/mL]"].values[0])

    energy_grid = np.linspace(
        ed - e_range / 2,
        ed + e_range / 2,
        n_points,
    )
    predictions = []
    for energy in energy_grid:
        params["Energy Density [J/mL]"] = energy
        results = predict(params)
        results["energy density"] = energy
        predictions.append(results)

    df = pd.DataFrame(predictions)

    fig = px.scatter(
        df,
        x="energy density",
        y="mean_prediction",
        error_y="std_prediction",
        labels={
            "energy density": "energy density / mJ mL⁻¹",
            "mean_prediction": "size / nm",
        },
    )
    return fig
