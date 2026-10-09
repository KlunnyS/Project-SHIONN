#!/usr/bin/env -S fish --no-config
# Fish entry point; the Bash launcher owns the actual model-run defaults.

set -l script_dir (path dirname (status filename))
exec "$script_dir/run_model.sh" \
    $argv
