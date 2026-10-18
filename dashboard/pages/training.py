"""Controls for the real SHIONN frozen-split behavior-cloning trainer."""

from dash import dcc, html


def layout():
    return html.Main(className="shell", children=[
        html.Div(className="hero", children=[
            html.P("BEHAVIOR CLONING · FROZEN SPLIT", className="eyebrow"),
            html.H1("Train SHIONN"),
            html.P("Launch the repository trainer with the approved v3 split and selected correction caches."),
        ]),
        html.Div(className="notice", children=[
            html.Strong("Validation is fixed: "),
            "dataset_test11 and dataset_test12 stay held out. The interface never adds validation/evaluation episodes to training.",
        ]),
        html.Section(className="panel", children=[
            html.H2("Run configuration"),
            html.Div(className="settings", children=[
                html.Label(["Epoch limit", dcc.Input(id="train-epochs", type="number", value=20, min=1, max=200, step=1)]),
                html.Label(["Device", dcc.Dropdown(id="train-device", options=[{"label": "Automatic", "value": "auto"}, {"label": "CUDA", "value": "cuda"}, {"label": "CPU", "value": "cpu"}], value="auto", clearable=False)]),
                html.Label(["Correction sampling share", dcc.Input(id="train-correction-fraction", type="number", value=0.1, min=0.01, max=0.5, step=0.01)]),
                html.Label(["New run name", dcc.Input(id="train-run-name", type="text", value="v5_recovery_proactive_v3", maxLength=64)]),
            ]),
            html.Label("Training correction data"),
            html.Ul([html.Li("31 approved recovery episodes"), html.Li("20 approved proactive-turn episodes")], className="details"),
            html.P("Sampling share is the expected fraction of optimizer draws allocated across the selected correction caches.", className="hint"),
            html.Button("Start training", id="start-training", className="button primary"),
            html.Div(id="train-launch-status", className="status"),
        ]),
        html.Section(className="panel", children=[
            html.H2("What this runs"),
            html.Pre(id="train-command-preview", className="command"),
            html.P("Training writes best.pt, last.pt, metrics.jsonl, and checkpoint contracts into the new run directory. Existing directories are never overwritten.", className="hint"),
        ]),
    ])
