"""Dataset, proactive-turn, and supervised recovery recording controls."""

from dash import dcc, html
from backend.project import TRAINING_MAPS, checkpoints


def layout():
    checkpoint_options = checkpoints()
    return html.Main(className="shell", children=[
        html.Div(className="hero", children=[
            html.P("OPERATOR-LED CAPTURE", className="eyebrow"),
            html.H1("Record demonstrations"),
            html.P("Each mode launches the existing recorder. The game remains under your control; the dashboard does not synthesize gameplay inputs."),
        ]),
        html.Section(className="panel", children=[
            html.H2("Recording mode"),
            dcc.RadioItems(id="record-mode", options=[
                {"label": "Standard demonstrations", "value": "demonstration"},
                {"label": "Proactive turn correction", "value": "proactive_turn"},
                {"label": "Model-assisted recovery", "value": "recovery"},
            ], value="demonstration", className="radios"),
            html.Label("Training chambers only (validation/evaluation chambers are excluded)"),
            dcc.Dropdown(id="record-maps", options=[{"label": item, "value": item} for item in TRAINING_MAPS], value=["dataset_test1"], multi=True),
            html.Div(id="record-demo-options", children=[
                html.Div(className="settings", children=[
                    html.Label(["Successful clips per chamber", dcc.Input(id="record-episodes", type="number", value=1, min=1, max=100, step=1)]),
                    html.Label(["Clip duration limit (seconds)", dcc.Input(id="record-duration", type="number", value=60, min=1, max=600, step=1)]),
                    html.Label(["Turn direction", dcc.Dropdown(id="turn-direction", options=[{"label": "Left", "value": "left"}, {"label": "Right", "value": "right"}], value="left", clearable=False)]),
                ]),
                html.P("For proactive mode, position before wall contact; physical P starts capture. Standard clips start when the chamber is ready.", className="hint"),
            ]),
            html.Div(id="record-recovery-options", style={"display": "none"}, children=[
                html.Label("Checkpoint for model-assisted recovery"),
                dcc.Dropdown(id="recovery-checkpoint", options=checkpoint_options, placeholder="Select a checkpoint", clearable=False),
                html.Div(className="settings", children=[
                    html.Label(["Accepted corrections target", dcc.Input(id="recovery-target", type="number", value=5, min=1, max=1000, step=1)]),
                    dcc.Checklist(id="recovery-test-mode", options=[{"label": " Test mode (do not save correction clips)", "value": "test"}], value=[]),
                ]),
                html.P("P hands over control, P starts a correction after mouse verification, R retries the chamber, K ends the session. Esc is emergency stop.", className="hint"),
            ]),
            html.Button("Start recorder", id="start-recording", className="button primary"),
            html.Div(id="record-launch-status", className="status"),
            html.Div(className="notice warning", children="Starting a recorder launches Portal 2 and may capture keyboard/mouse input. Stop the active job from the task panel or use its documented physical emergency key."),
        ]),
    ])
