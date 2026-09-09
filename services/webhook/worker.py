import httpx
from celery import Celery
import os

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("webhook", broker=redis_url, backend=redis_url)
app.conf.task_routes = {
    "analyze_pr": {"queue": "webhook"},
    "trigger_learning": {"queue": "learning"},
}


@app.task(name="analyze_pr")
def analyze_pr(
    pr_full_name,
    pr_number,
    head_sha,
    installation_id,
    pull_request_id,
):
    with httpx.Client() as client:
        response = client.post(
            "http://orchestrator:8002/analyze",
            json={
                "pr_id": str(pull_request_id),
                "pr_number": pr_number,
                "repo_full_name": pr_full_name,
                "head_sha": head_sha,
                "installation_id": installation_id,
            },
            timeout=120,
        )
        response.raise_for_status()


@app.task(name="trigger_learning")
def trigger_learning(repo_full_name, pull_request_id):
    with httpx.Client() as client:
        response = client.post(
            "http://learner:8004/learn",
            json={
                "repo_full_name": repo_full_name,
                "pull_request_id": pull_request_id
            },
            timeout=60
        )
