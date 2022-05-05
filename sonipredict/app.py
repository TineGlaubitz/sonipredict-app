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
    "concentration": 3.6,
    "volume": 5.2,
    "energy_density": 2000,
    "energy": 4347,
    "ed_range": 1000,
    "ed_points": 20,
    "iep": 2.5,
    "zeta": -26,
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
                                    dbc.Col(
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
                                    dbc.Col(
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
                                    dbc.Col(
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
                                    dbc.Col(
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
                                                    for i in [500, 1_000, 5_000, 10_000]
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
                                    dbc.Col(
                                        [
                                            html.Label("zeta potential / mV"),
                                            dcc.Slider(
                                                id="zeta",
                                                min=-50,
                                                max=50,
                                                step=1,
                                                value=DEFAULTS["zeta"],
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in [-50, -25, 0, 25, 50]
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
                                    dbc.Col(
                                        [
                                            html.Label("total energy / J"),
                                            dcc.Slider(
                                                id="energy",
                                                min=0,
                                                max=5000,
                                                step=500,
                                                value=DEFAULTS["energy"],
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in [
                                                        0,
                                                        2000,
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
                                    dbc.Col(
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
                                    dbc.Col(
                                        [
                                            html.Label("volume / mL"),
                                            dcc.Slider(
                                                id="volume",
                                                min=0,
                                                max=50,
                                                step=5,
                                                value=DEFAULTS["volume"],
                                                marks={
                                                    i: "{}".format(i)
                                                    for i in [
                                                        0,
                                                        25,
                                                        50,
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
                                    dbc.Col(
                                        [
                                            html.Label("particle type"),
                                            dcc.Dropdown(
                                                options=[
                                                    {
                                                        "label": "silica",
                                                        "value": "SiO2",
                                                    },
                                                    {
                                                        "label": "titania",
                                                        "value": "TiO2",
                                                    },
                                                    {
                                                        "label": "cerium oxide",
                                                        "value": "CeO2",
                                                    },
                                                    {
                                                        "label": "zink oxide    ",
                                                        "value": "ZnO",
                                                    },
                                                ],
                                                value=DEFAULTS["particle_type"],
                                                id="particle",
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
                html.Hr(),
                html.H2("Prediction"),
                html.P(id="prediction_str"),
            ],
            className="container",
        ),
        html.Div(
            [
                html.Hr(),
                html.H2("Energy density dependence"),
                html.P(),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    dbc.Col(
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
                                                    for i in [100, 1_000, 10_000]
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
                                    dbc.Col(
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
            ],
            className="container",
        ),
        html.Hr(),
        html.Footer("© 2022, Tine Glaubitz. Web app version {}".format(__version__)),
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
        Input("energy", "value"),
        Input("coating", "value"),
        Input("volume", "value"),
        Input("particle", "value"),
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
    energy,
    coating,
    volume,
    particle_type,
    ed_range,
    ed_points,
):
    # ['Size PP [nm]',
    # 'Concentration [mg/mL]',
    # 'Energy Density [J/mL]',
    # 'Isoelectric Point',
    # 'Zeta Pot[mV]',
    # 'Volume [mL]',
    # 'Total Energy [J]',
    # 'Coating',
    # 'Particle']
    values = np.array(
        [
            particle_size,
            concentration,
            energy_density,
            iep,
            zeta,
            volume,
            energy,
            coating,
            particle_type,
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
