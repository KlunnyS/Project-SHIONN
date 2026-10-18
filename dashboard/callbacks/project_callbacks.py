"""Callbacks for SHIONN status, task submission, and monitored process output."""

import json
import shlex

import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, html, no_update

from backend.project import (
    LIVE_BENCHMARK_MAPS,
    active_job,
    benchmark_history,
    build_command,
    project_status,
    start_task,
    stop_task,
    training_run_data,
)


def _card(label, value, note=""):
    return html.Div(className="metric", children=[
        html.Small(label), html.Strong(str(value)), html.Span(note, className="metric-note"),
    ])


@callback(
    Output("status-cards", "children"),
    Output("split-summary", "children"),
    Output("checkpoint-summary", "children"),
    Input("project-poll", "n_intervals"),
)
def refresh_project_status(_tick):
    status = project_status()
    groups = status.get("groups", {})
    cache_state = status.get("cache_dirs", {})
    cache_text = " · ".join(
        f"{name}: {'folder present' if cache_state.get(name) else 'missing'}"
        for name in ("base", "recovery", "proactive_turn")
    )
    cards = [
        _card("Frozen split", status.get("split_id", "missing"), "approved episode manifest"),
        _card("Training", status.get("train", 0), "episodes"),
        _card("Validation", status.get("validation", 0), "episodes · whole chambers"),
        _card("Checkpoints", len(status.get("checkpoints", [])), "with JSON contracts"),
    ]
    split = html.Div(children=[
        html.P(f"{status.get('split_id', 'missing')} · {status.get('train', 0)} train / {status.get('validation', 0)} validation / {status.get('evaluation', 0)} cached evaluation"),
        html.P("Cache groups: " + ", ".join(f"{name} {count}" for name, count in sorted(groups.items()))),
        html.P(cache_text, className="hint"),
        html.P("Validation: dataset_test11 and dataset_test12 · live evaluation: evaluation1 and evaluation2", className="hint"),
    ])
    checkpoints_summary = html.Div(children=[
        html.P("No valid best.pt checkpoints discovered." if not status.get("checkpoints") else "Available models:"),
        html.Ul([html.Li(item["label"]) for item in status.get("checkpoints", [])[:8]]),
    ])
    return cards, split, checkpoints_summary


@callback(
    Output("record-demo-options", "style"),
    Output("record-recovery-options", "style"),
    Input("record-mode", "value"),
)
def show_recording_options(mode):
    return ({"display": "none"} if mode == "recovery" else {},
            {} if mode == "recovery" else {"display": "none"})


@callback(
    Output("live-options", "style"),
    Output("benchmark-maps", "options"),
    Output("benchmark-maps", "value"),
    Input("benchmark-mode", "value"),
    State("benchmark-maps", "value"),
)
def show_benchmark_options(mode, current):
    if mode == "live":
        options = LIVE_BENCHMARK_MAPS
        selected = [name for name in (current or []) if name in options] or ["evaluation1", "evaluation2"]
        return {}, [{"label": item, "value": item} for item in options], selected
    options = ["dataset_test11", "dataset_test12"]
    selected = [name for name in (current or []) if name in options] or options
    return {"display": "none"}, [{"label": item, "value": item} for item in options], selected


@callback(
    Output("train-command-preview", "children"),
    Input("train-epochs", "value"),
    Input("train-device", "value"),
    Input("train-correction-fraction", "value"),
    Input("train-run-name", "value"),
)
def preview_training_command(epochs, device, fraction, run_name):
    try:
        command, _ = build_command("train", {
            "epochs": epochs or 20, "device": device or "auto",
            "correction_fraction": fraction or 0.1,
            "run_name": run_name or "v5_recovery_proactive_v3",
        })
        return shlex.join(command)
    except (TypeError, ValueError) as exc:
        return "Configuration needs attention: " + str(exc)


@callback(
    Output("active-task", "data"),
    Output("record-launch-status", "children"),
    Output("benchmark-launch-status", "children"),
    Output("train-launch-status", "children"),
    Input("start-recording", "n_clicks"),
    Input("start-benchmark", "n_clicks"),
    Input("start-training", "n_clicks"),
    State("record-mode", "value"),
    State("record-maps", "value"),
    State("record-episodes", "value"),
    State("record-duration", "value"),
    State("turn-direction", "value"),
    State("recovery-checkpoint", "value"),
    State("recovery-target", "value"),
    State("recovery-test-mode", "value"),
    State("benchmark-mode", "value"),
    State("benchmark-checkpoint", "value"),
    State("benchmark-maps", "value"),
    State("benchmark-repeats", "value"),
    State("benchmark-seconds", "value"),
    State("benchmark-device", "value"),
    State("benchmark-dry-run", "value"),
    State("train-epochs", "value"),
    State("train-device", "value"),
    State("train-correction-fraction", "value"),
    State("train-run-name", "value"),
    prevent_initial_call=True,
)
def launch_requested_task(
    _record_click, _benchmark_click, _train_click,
    record_mode, record_maps, record_episodes, record_duration, turn_direction,
    recovery_checkpoint, recovery_target, recovery_test_mode,
    benchmark_mode, benchmark_checkpoint, benchmark_maps, benchmark_repeats,
    benchmark_seconds, benchmark_device, benchmark_dry_run,
    train_epochs, train_device, train_fraction, train_run_name,
):
    task_id = ctx.triggered_id
    try:
        if task_id == "start-recording":
            config = {
                "mode": record_mode, "maps": record_maps,
                "episodes": record_episodes, "duration": record_duration,
                "direction": turn_direction, "checkpoint": recovery_checkpoint,
                "target_corrections": recovery_target,
                "test_mode": "test" in (recovery_test_mode or []),
            }
            task_type = "record"
            status_target = "record"
        elif task_id == "start-benchmark":
            config = {
                "mode": benchmark_mode, "checkpoint": benchmark_checkpoint,
                "maps": benchmark_maps, "repeats": benchmark_repeats,
                "max_seconds": benchmark_seconds, "device": benchmark_device,
                "dry_run": "dry" in (benchmark_dry_run or []),
            }
            task_type = "benchmark"
            status_target = "benchmark"
        elif task_id == "start-training":
            config = {
                "epochs": train_epochs, "device": train_device,
                "correction_fraction": train_fraction,
                "run_name": train_run_name,
            }
            task_type = "train"
            status_target = "train"
        else:
            return no_update, no_update, no_update, no_update
        job = start_task(task_type, config)
        message = html.Span(f"Started {job['label']} · task {job['id']}", className="success")
        values = [no_update, no_update, no_update]
        values[{"record": 0, "benchmark": 1, "train": 2}[status_target]] = message
        return job, *values
    except Exception as exc:
        message = html.Span(str(exc), className="error")
        values = [no_update, no_update, no_update]
        target = {"start-recording": 0, "start-benchmark": 1, "start-training": 2}.get(task_id)
        if target is not None:
            values[target] = message
        return no_update, *values


@callback(
    Output("task-state", "children"),
    Output("task-output", "children"),
    Output("stop-task", "disabled"),
    Input("project-poll", "n_intervals"),
)
def poll_active_task(_tick):
    job = active_job()
    if not job:
        return "No dashboard task launched", "", True
    state_class = "task-state " + job["state"].lower()
    details = f"{job['label']} · {job['state']} · {job['task_dir']}"
    if job.get("returncode") is not None:
        details += f" · exit {job['returncode']}"
    output = html.Div(children=[
        html.P(job.get("last_line") or "Waiting for subprocess output…", className=state_class),
        html.Details([html.Summary("Recent output"), html.Pre(job.get("output", ""), className="log-output")]),
    ])
    stoppable = job["state"] in {"Running", "Starting"}
    return details, output, not stoppable


@callback(
    Output("task-state", "children", allow_duplicate=True),
    Input("stop-task", "n_clicks"),
    prevent_initial_call=True,
)
def stop_active_task(_clicks):
    job = stop_task()
    if not job:
        return "No active task."
    return f"Stop requested for {job.get('label', 'task')}; waiting for a graceful exit."


@callback(
    Output("result-training-run", "options"),
    Output("benchmark-history", "children"),
    Input("project-poll", "n_intervals"),
)
def populate_results(_tick):
    from backend.project import training_run_options, benchmark_history
    runs = training_run_options()
    history = benchmark_history()
    if not history:
        rendered = html.P("No saved live sequence summaries found.")
    else:
        rendered = html.Ul([
            html.Li(f"{item['path']} · {json.dumps(item['summary'], ensure_ascii=False)[:500]}")
            for item in history
        ])
    return runs, rendered


@callback(
    Output("training-result-summary", "children"),
    Output("training-result-chart", "figure"),
    Input("result-training-run", "value"),
    prevent_initial_call=False,
)
def show_training_run(run_name):
    figure = go.Figure()
    if not run_name:
        return html.P("Choose a run with a metrics.jsonl file."), figure
    try:
        metadata, rows = training_run_data(run_name)
        config = metadata.get("config", {})
        if rows:
            epochs = [row.get("epoch") for row in rows]
            figure.add_trace(go.Scatter(x=epochs, y=[row.get("train", {}).get("total") for row in rows], name="Train loss"))
            figure.add_trace(go.Scatter(x=epochs, y=[row.get("validation", {}).get("total") for row in rows], name="Validation loss"))
            figure.add_trace(go.Scatter(x=epochs, y=[row.get("validation", {}).get("mouse") for row in rows], name="Validation mouse loss", yaxis="y2"))
        figure.update_layout(template="plotly_dark", title=f"{run_name} · epoch metrics",
                             paper_bgcolor="#131a29", plot_bgcolor="#131a29",
                             xaxis_title="Epoch", yaxis_title="Loss",
                             yaxis2={"title": "Mouse loss", "overlaying": "y", "side": "right"})
        best = metadata.get("best_val_loss", "not saved")
        summary = html.Div([
            html.P(f"{len(rows)} completed epochs · best checkpoint validation loss {best}"),
            html.P(f"Architecture: {config.get('architecture', 'unknown')} · split: {config.get('split_id', 'see checkpoint config')}"),
        ])
        return summary, figure
    except Exception as exc:
        return html.P("Could not read this training run: " + str(exc), className="error"), figure
