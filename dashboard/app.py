"""SHIONN local operations dashboard. Run with ``python app.py``."""

from dash import Dash, Input, Output, dcc, html

from pages.benchmark import layout as benchmark_layout
from pages.overview import layout as overview_layout
from pages.recording import layout as recording_layout
from pages.results import layout as results_layout
from pages.training import layout as training_layout

import callbacks.project_callbacks  # noqa: E402,F401


app = Dash(__name__, suppress_callback_exceptions=True, title="SHIONN Control Room")

app.layout = html.Div(children=[
    dcc.Location(id="url"),
    dcc.Interval(id="project-poll", interval=1500, n_intervals=0),
    dcc.Store(id="active-task"),
    html.Header(className="topbar", children=[
        html.Div([html.Span("SHIONN", className="brand-mark"), html.Strong("Control Room")]),
        html.Nav(children=[
            dcc.Link("Overview", href="/", className="nav-link"),
            dcc.Link("Record", href="/record", className="nav-link"),
            dcc.Link("Benchmark", href="/benchmark", className="nav-link"),
            dcc.Link("Train", href="/train", className="nav-link"),
            dcc.Link("Results", href="/results", className="nav-link"),
        ]),
    ]),
    html.Section(className="task-bar", children=[
        html.Div(id="task-state", children="No dashboard task launched"),
        html.Button("Stop active task", id="stop-task", className="button danger", disabled=True),
        html.Div(id="task-output", className="task-output"),
    ]),
    html.Div(id="page-content"),
])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(pathname):
    if pathname == "/record":
        return recording_layout()
    if pathname == "/benchmark":
        return benchmark_layout()
    if pathname == "/train":
        return training_layout()
    if pathname == "/results":
        return results_layout()
    return overview_layout()


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
