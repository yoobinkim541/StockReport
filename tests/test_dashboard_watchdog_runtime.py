import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _sandbox(tmp_path, worker, dependency_path, entrypoint):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for name in ("dashboard", "ml", "providers", "agent_console", "bot", "lib", "config", "reports"):
        (project / name).mkdir()
    (project / ".env").write_text("DASHBOARD_ENABLED=true\n", encoding="utf-8")
    launch_log = tmp_path / "launch.log"
    script = scripts / "dashboard_watchdog.sh"
    body = (ROOT / "scripts/dashboard_watchdog.sh").read_text()
    body = body.replace('LOG_FILE="/tmp/dashboard.log"', f'LOG_FILE="{tmp_path / "dashboard.log"}"')
    body = body.replace('WATCHDOG_LOCK="/tmp/dashboard_watchdog.lock"',
                        f'WATCHDOG_LOCK="{tmp_path / "watchdog.lock"}"')
    script.write_text(body)
    (scripts / "run_dashboard.sh").write_text(
        '#!/usr/bin/env bash\nprintf "started\\n" >> "$DASHBOARD_TEST_LAUNCH_LOG"\n')
    bins = tmp_path / "bin"
    bins.mkdir()
    for name, content in {
        "curl": '#!/usr/bin/env bash\nprintf "ok\\n"\n',
        "ps": (
            '#!/usr/bin/env bash\n'
            'if [[ "$1" == "-eo" ]]; then\n'
            '  printf "  2147483647 bash -c .venv/bin/streamlit run dashboard/server.py\\n"\n'
            f'  printf "  {worker.pid} /project/.venv/bin/python3 .venv/bin/streamlit run dashboard/{entrypoint}\\n"\n'
            'else\n'
            f'  exec "{shutil.which("ps")}" "$@"\n'
            'fi\n'
        ),
    }.items():
        path = bins / name
        path.write_text(content)
        path.chmod(0o755)
    dependency = project / dependency_path
    dependency.write_text("# test source\n")
    env = {**os.environ, "PATH": str(bins) + os.pathsep + os.environ["PATH"],
           "DASHBOARD_TEST_LAUNCH_LOG": str(launch_log)}
    return script, dependency, launch_log, env


@pytest.mark.parametrize(("dependency_path", "entrypoint"), [
    ("reports/wiki_distillation.py", "app.py"),
    ("dashboard/reconnect.js", "server.py"),
])
def test_newer_dependency_restarts_the_exact_worker(tmp_path, dependency_path, entrypoint):
    worker = subprocess.Popen(["sleep", "300"])
    try:
        script, dependency, launch_log, env = _sandbox(tmp_path, worker, dependency_path, entrypoint)
        future = time.time() + 60
        os.utime(dependency, (future, future))
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert worker.poll() is not None
        deadline = time.monotonic() + 3
        while not launch_log.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert launch_log.read_text() == "started\n"
    finally:
        if worker.poll() is None:
            worker.terminate()
        worker.wait(timeout=5)


def test_older_reports_do_not_restart_a_healthy_dashboard_worker(tmp_path):
    worker = subprocess.Popen(["sleep", "300"])
    try:
        script, dependency, launch_log, env = _sandbox(tmp_path, worker, "dashboard/reconnect.js", "server.py")
        old = time.time() - 300
        os.utime(dependency, (old, old))
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert worker.poll() is None
        assert not launch_log.exists()
    finally:
        if worker.poll() is None:
            worker.terminate()
        worker.wait(timeout=5)
