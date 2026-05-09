from __future__ import annotations

import json
import subprocess
import threading
import time
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
REFRESH_LOCK = threading.Lock()
REFRESH_COMMANDS = [
    ["python3", str(BASE_DIR / "commodity_macro_dashboard.py")],
    ["python3", str(BASE_DIR / "market_daily_charts.py")],
    ["python3", str(BASE_DIR / "us_stock_indicator_pipeline.py")],
    ["python3", str(BASE_DIR / "build_market_monitor_dashboard.py")],
    ["python3", str(BASE_DIR / "build_strategy_dashboard.py")],
]


def run_refresh_pipeline() -> dict:
    logs = []
    started_at = time.time()
    for command in REFRESH_COMMANDS:
        label = Path(command[-1]).name
        try:
            result = subprocess.run(
                command,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:  # noqa: PERF203
            logs.append({"step": label, "ok": False, "exit_code": None, "output": str(exc)})
            return {
                "ok": False,
                "logs": logs,
                "duration_seconds": round(time.time() - started_at, 2),
                "message": f"{label} 执行异常",
            }

        combined_output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
        logs.append(
            {
                "step": label,
                "ok": result.returncode == 0,
                "exit_code": result.returncode,
                "output": combined_output,
            }
        )
        if result.returncode != 0:
            return {
                "ok": False,
                "logs": logs,
                "duration_seconds": round(time.time() - started_at, 2),
                "message": f"{label} 刷新失败",
            }

    return {
        "ok": True,
        "logs": logs,
        "duration_seconds": round(time.time() - started_at, 2),
        "message": "刷新完成",
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/refresh":
            self._write_json({"ok": False, "message": "Not Found"}, status=HTTPStatus.NOT_FOUND)
            return

        acquired = REFRESH_LOCK.acquire(blocking=False)
        if not acquired:
            self._write_json({"ok": False, "message": "已有刷新任务正在执行"}, status=HTTPStatus.CONFLICT)
            return

        try:
            result = run_refresh_pipeline()
        finally:
            REFRESH_LOCK.release()

        status = HTTPStatus.OK if result.get("ok") else HTTPStatus.INTERNAL_SERVER_ERROR
        self._write_json(result, status=status)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), partial(DashboardHandler))
    print(f"Serving dashboard at http://{HOST}:{PORT}/output/market_monitor_dashboard.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
