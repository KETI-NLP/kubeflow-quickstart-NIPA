import argparse
import json
import subprocess
import time


def run_command(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def get_pod_json(namespace, pod_name):
    result = run_command(["kubectl", "get", "pod", "-n", namespace, pod_name, "-o", "json"])
    return json.loads(result.stdout)


def get_container_reason(pod_json):
    statuses = pod_json.get("status", {}).get("containerStatuses") or []
    if not statuses:
        return "unknown"
    state = statuses[0].get("state", {})
    terminated = state.get("terminated")
    if terminated:
        reason = terminated.get("reason", "Terminated")
        exit_code = terminated.get("exitCode")
        return f"{reason} (exit_code={exit_code})"
    waiting = state.get("waiting")
    if waiting:
        return waiting.get("reason", "Waiting")
    return "running"


def send_telegram(message):
    cmd = [
        "python",
        "/home/user/telegram_bot/send_telegram_message.py",
        "text",
        f"[작업 알림] {message}",
    ]
    run_command(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pod_name", required=True)
    parser.add_argument("--label", default="작업")
    parser.add_argument("--poll_seconds", type=int, default=30)
    args = parser.parse_args()

    while True:
        pod_json = get_pod_json(args.namespace, args.pod_name)
        phase = pod_json.get("status", {}).get("phase", "Unknown")
        reason = get_container_reason(pod_json)
        print(
            f"[Watcher] pod={args.pod_name} | phase={phase} | reason={reason}",
            flush=True,
        )

        if phase == "Succeeded":
            send_telegram(f"{args.label} 완료: pod={args.pod_name} | phase={phase}")
            return

        if phase == "Failed":
            send_telegram(f"{args.label} 실패: pod={args.pod_name} | phase={phase} | reason={reason}")
            return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
