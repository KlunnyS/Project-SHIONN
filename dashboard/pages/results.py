"""Persisted SHIONN checkpoint training and live benchmark results."""

from dash import dcc, html
from backend.project import training_run_options


def layout():
    return html.Main(className="shell", children=[
        html.Div(className="hero", children=[
            html.P("LOCAL RUN ARCHIVE", className="eyebrow"),
            html.H1("Training and benchmark results"),
            html.P("Training curves are read from the project's metrics.jsonl; live summaries come from sequence manifests."),
        ]),
        html.Section(className="panel", children=[
            html.H2("Training run"),
            dcc.Dropdown(id="result-training-run", options=training_run_options(), placeholder="Select a checkpoint run", clearable=False),
            html.Div(id="training-result-summary", className="details"),
            dcc.Graph(id="training-result-chart"),
        ]),
        html.Section(className="panel", children=[
            html.H2("Recent live sequences"),
            html.Div(id="benchmark-history", className="details"),
        ]),
    ])
