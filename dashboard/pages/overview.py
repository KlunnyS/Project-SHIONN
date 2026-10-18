"""Project status landing page."""

from dash import dcc, html


def layout():
    return html.Main(className="shell", children=[
        html.Div(className="hero", children=[
            html.P("PORTAL 2 · VISION-ONLY AGENT", className="eyebrow"),
            html.H1("SHIONN Control Room"),
            html.P("Local controls for data collection, model training, and held-out evaluation."),
        ]),
        html.Div(id="status-cards", className="metrics status-cards"),
        html.Div(className="grid-two", children=[
            html.Section(className="panel", children=[
                html.H2("Dataset split"), html.Div(id="split-summary", className="details"),
                html.P("Frozen v3 keeps dataset_test11/12 for validation; evaluation1/2 are not cached for training.", className="hint"),
            ]),
            html.Section(className="panel", children=[
                html.H2("Local checkpoints"), html.Div(id="checkpoint-summary", className="details"),
                html.P("Only checkpoints with their JSON configuration contract are selectable.", className="hint"),
            ]),
        ]),
        html.Section(className="panel", children=[
            html.H2("Quick workflow"),
            html.Div(className="quick-links", children=[
                dcc.Link("Record demonstrations →", href="/record", className="quick-link"),
                dcc.Link("Benchmark a checkpoint →", href="/benchmark", className="quick-link"),
                dcc.Link("Train on frozen v3 →", href="/train", className="quick-link"),
                dcc.Link("Inspect previous runs →", href="/results", className="quick-link"),
            ]),
        ]),
    ])
