from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker

from models import PullRequest, Settings
from worker import analyze_pr, trigger_learning

settings = Settings()
engine = create_async_engine(settings.database_url)
AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession)

app = FastAPI()
Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/events", status_code=202)
async def receive_event(request: Request):
    body = await request.json()
    action = body.get("action", "")
    pull_request = body.get("pull_request", {})

    if action == "closed" and pull_request.get("merged"):
        pr_number = pull_request.get("number")
        repo_full_name = body.get("repository", {}).get("full_name")
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(PullRequest).where(
                    PullRequest.repo_full_name == repo_full_name,
                    PullRequest.pr_number == pr_number
                )
            )
            pr_record = result.scalar_one_or_none()
            if pr_record:
                trigger_learning.apply_async(
                    args=[repo_full_name, str(pr_record.id)], queue="learning")
        return {"status": "accepted"}

    if action not in ["opened", "synchronize", "reopened"]:
        return {"status": "skipped"}

    pr_number = pull_request.get("number")
    repo_full_name = body.get("repository", {}).get("full_name")
    head_sha = pull_request.get("head", {}).get("sha", "")
    installation_id = body.get("installation", {}).get("id", 0)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(PullRequest).where(
                PullRequest.repo_full_name == repo_full_name,
                PullRequest.pr_number == pr_number
            )
        )
        if result.scalar_one_or_none():
            return {"status": "already processing"}

        pull_request_record = PullRequest(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
            installation_id=installation_id,
            status="pending"
        )
        session.add(pull_request_record)
        await session.commit()
        await session.refresh(pull_request_record)
        pull_request_id = str(pull_request_record.id)

        analyze_pr.apply_async(
            args=[repo_full_name, pr_number, head_sha,
                  installation_id, pull_request_id],
            queue="webhook"
        )
        return {"status": "accepted"}
