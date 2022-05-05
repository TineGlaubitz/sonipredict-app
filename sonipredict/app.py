import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy as np
from . import __version__, FEATURES
from .core import get_pca_plot, get_isomap_plot, make_prediction, run_energy_sweep
import pandas as pd

EXTERNAL_STYLESHEETS = [
    "./assets/style.css",
    "./assets/vis.min.css",
    dbc.themes.MINTY,
]


dash_app = dash.Dash(  # pylint:disable=invalid-name
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    meta_tags=[
        {"charset": "utf-8"},
        {"http-equiv": "X-UA-Compatible", "content": "IE=edge"},
        # needed for iframe resizer
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

server = dash_app.server  # pylint:disable=invalid-name
dash_app.title = "sonipredict"


DEFAULTS = {
    "size": 60,
    "concentration": 5,
    "volume": 5,
    "energy_density": 500,
    "ed_range": 1000,
    "ed_points": 20,
    "bet": 100,
    "iep": 2,
    "zeta": -40,
    "coating": "Hydrophil",
    "particle_type": "SiO2",
}


layout = html.Div(
    [
        html.Div(
            [
                html.Img(src="assets/logo.svg", width="400px"),
                html.P(
                    "Predict the outcome of dispersion experiments.",
                    className="lead",
                ),
            ],
            className="jumbotron",
            style={
                "margin-bottom": "1rem",
                "padding-bottom": "2rem",
                "text-align": "center",
            },
        ),
        html.Div(
            [
                html.H2("Sonication parameters"),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("particle size / nm"),
                                            dcc.Slider(
                                                id="size",
                                                min=20,
                                                max=100,
                                                value=DEFAULTS["size"],
                                                step=5,
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in range(20, 100, 20)
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("concentration / mg/mL"),
                                            dcc.Slider(
                                                id="concentration",
                                                min=1,
                                                max=10,
                                                value=DEFAULTS["concentration"],
                                                step=0.1,
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in range(1, 10, 2)
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("isoelectric point"),
                                            dcc.Slider(
                                                id="iep",
                                                min=1,
                                                max=12,
                                                value=DEFAULTS["iep"],
                                                step=0.5,
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in range(1, 10, 2)
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("energy density / J/mL"),
                                            dcc.Slider(
                                                id="energy_density",
                                                min=100,
                                                max=10_000,
                                                step=0.5,
                                                value=DEFAULTS["energy_density"],
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in [100, 1_000, 5_000, 10_000]
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("zeta potential / mV"),
                                            dcc.Slider(
                                                id="zeta",
                                                min=-20,
                                                max=20,
                                                step=0.5,
                                                value=DEFAULTS["zeta"],
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in [-7, -4, -2, 0, 2, 4, 7]
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("BET surface area"),
                                            dcc.Slider(
                                                id="bet",
                                                min=0,
                                                max=500,
                                                step=0.5,
                                                value=DEFAULTS["bet"],
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in [
                                                        0,
                                                        100,
                                                        200,
                                                        300,
                                                        400,
                                                        5000,
                                                    ]
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("coating"),
                                            dcc.Dropdown(
                                                options=[
                                                    {
                                                        "label": "hydrophylic",
                                                        "value": "Hydrophil",
                                                    },
                                                    {
                                                        "label": "hydrophobic",
                                                        "value": "Hydrophob",
                                                    },
                                                ],
                                                value=DEFAULTS["coating"],
                                                id="coating",
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("energy density range"),
                                            dcc.Slider(
                                                id="ed_range",
                                                min=10,
                                                max=10000,
                                                step=0.5,
                                                value=DEFAULTS["ed_range"],
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in [10, 100, 1_000, 10_000]
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            html.Label("number energy density points"),
                                            dcc.Slider(
                                                id="ed_points",
                                                min=5,
                                                max=60,
                                                step=1,
                                                value=DEFAULTS["ed_points"],
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in range(5, 60, 10)
                                                },
                                            ),
                                        ]
                                    )
                                ]
                            )
                        ),
                    ]
                ),
            ],
            className="container",
        ),
        html.Div(
            [
                html.H2("Prediction"),
                html.P(id="prediction_str"),
            ],
            className="container",
        ),
        html.Div(
            [
                html.H2("Energy density dependence"),
                html.P(
                    "The plot below show the particle size as function of the energy density."
                ),
                dcc.Graph(id="ed_plot"),
            ],
            className="container",
        ),
        html.Div(
            [
                html.H2("Visualization of the input"),
                html.P(
                    [
                        "The plot below shows a ",
                        html.A("ISOMAP ", href="https://en.wikipedia.org/wiki/Isomap"),
                        "projection of the training data of the model and where the query point lies in this space.",
                    ]
                ),
                dcc.Graph(id="isomap"),
                html.P(
                    [
                        "The plot below shows the training data of the model in the space of the first two ",
                        html.A(
                            "principal components ",
                            href="https://en.wikipedia.org/wiki/Principal_component_analysis",
                        ),
                        "and where the query point lies in this space.",
                    ]
                ),
                dcc.Graph(id="pca"),
            ],
            className="container",
        ),
        html.Div(
            [
                html.H2("About"),
                html.P(
                    "This app can provide provide guidance in selecting the dispersion parameters for nanoparticles. The model that is currently deployed in this app was trained on data we obtained for silica nanoparticles synthesized in our lab, for which we estimated the size using dynamic light scattering. To this dataset we added data we found by mining the literature."
                ),
                html.P(
                    "The model is an ensemble of gradient boosting regressors that were trained with slighly different inputs and hence can be used as to stabilize the variance and reduce the uncertainty."
                ),
                html.P(
                    "You should not trust the predictions of the model if the red dot (for the query point) is far from the gray points in the ISOMAP and PCA plot. This means that the parameters you entered are very different from the ones the model was trained on."
                ),
                html.P(
                    "In the current implementation the model has no understanding of the chemistry as the particles types are encoded using one-hot encoding."
                ),
            ],
            className="container",
        ),
        html.Hr(),
        html.Footer("© 2021, Tine Glaubitz. Web app version {}".format(__version__)),
    ]
)

dash_app.layout = layout


@dash_app.callback(
    [
        Output("isomap", "figure"),
        Output("pca", "figure"),
        Output("ed_plot", "figure"),
        Output("prediction_str", "children"),
    ],
    [
        Input("size", "value"),
        Input("concentration", "value"),
        Input("iep", "value"),
        Input("energy_density", "value"),
        Input("zeta", "value"),
        Input("bet", "value"),
        Input("coating", "value"),
        Input("ed_range", "value"),
        Input("ed_points", "value"),
    ],
)
def update_figure(
    particle_size,
    concentration,
    iep,
    energy_density,
    zeta,
    bet,
    coating,
    ed_range,
    ed_points,
):
    #             "Size PP [nm]",
    #     "Concentration [mg/mL]",
    #     "Energy Density [J/mL]",
    #     "abs_zeta",
    #     "iep_stability",
    #     "BET [m2/g]",
    #     "Isoelectric Point",
    #     "Zeta Pot[mV]",
    #     COATING
    values = np.array(
        [
            particle_size,
            concentration,
            energy_density,
            np.abs(zeta),
            np.abs(iep - 7),
            bet,
            iep,
            zeta,
            coating,
        ]
    )

    df = pd.DataFrame([dict(zip(FEATURES, values))])

    isomap = get_isomap_plot(df)
    pca = get_pca_plot(df)
    prediction = make_prediction(df)
    ed_sweep = run_energy_sweep(df, float(ed_range), int(ed_points))
    return isomap, pca, ed_sweep, prediction


if __name__ == "__main__":
    dash_app.run_server(debug=True)
