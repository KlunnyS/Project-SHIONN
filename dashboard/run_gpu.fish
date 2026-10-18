#!/usr/bin/env fish

set project_dir (dirname (status --current-filename))
cd $project_dir

set python_path .venv-gpu/bin/python
if not test -x $python_path
    echo "GPU environment not found. Run ./setup_gpu.fish first."
    exit 1
end

$python_path scripts/check_gpu.py
or exit 1

exec $python_path app.py
