"""Offline validation and live held-out benchmark controls."""

from dash import dcc, html
from backend.project import LIVE_BENCHMARK_MAPS, checkpoints


def layout():
    return html.Main(className="shell", children=[
        html.Div(className="hero", children=[
            html.P("HELD-OUT EVALUATION", className="eyebrow"),
            html.H1("Benchmark checkpoints"),
            html.P("Score actions on validation recordings or run an autonomous live sequence in Portal 2."),
        ]),
        html.Section(className="panel", children=[
            html.H2("Model and test"),
            html.Label("Checkpoint"),
            dcc.Dropdown(id="benchmark-checkpoint", options=checkpoints(), placeholder="Select a checkpoint", clearable=False),
            dcc.RadioItems(id="benchmark-mode", options=[
                {"label": "Offline expert comparison (no game controls)", "value": "expert"},
                {"label": "Live benchmark sequence", "value": "live"},
            ], value="expert", className="radios"),
            html.Label("Chambers"),
            dcc.Dropdown(id="benchmark-maps", options=[{"label": name, "value": name} for name in ["dataset_test11", "dataset_test12", "evaluation1", "evaluation2"]], value=["dataset_test11", "dataset_test12"], multi=True),
            html.Div(id="live-options", style={"display": "none"}, children=[
                html.Div(className="settings", children=[
                    html.Label(["Repeats per chamber", dcc.Input(id="benchmark-repeats", type="number", value=3, min=1, max=20, step=1)]),
                    html.Label(["Seconds per attempt", dcc.Input(id="benchmark-seconds", type="number", value=60, min=1, max=600, step=1)]),
                    html.Label(["Device", dcc.Dropdown(id="benchmark-device", options=[{"label": name, "value": name} for name in ["auto", "cuda", "cpu"]], value="auto", clearable=False)]),
                ]),
                dcc.Checklist(id="benchmark-dry-run", options=[{"label": " Dry-run (predict only; do not control the game)", "value": "dry"}], value=[]),
            ]),
            html.Button("Run benchmark", id="start-benchmark", className="button primary"),
            html.Div(id="benchmark-launch-status", className="status"),
            html.Div(className="notice warning", children="A live run sends model input to Portal 2 and may control the focused desktop. Start only when the game/compositor is ready; live mode is restricted to validation/evaluation chambers."),
        ]),
    ])
