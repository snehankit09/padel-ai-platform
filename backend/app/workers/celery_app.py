"""
Celery application instance.

Why Celery + Redis instead of just calling the pipeline in the request:
the PRD's NFRs require the system to scale GPU-bound stages independently
(Section 6, Scalability) and to survive partial failures without
reprocessing a whole video (Reliability). A task queue gives us both:
each pipeline stage becomes a task that can retry independently, run on
GPU worker machines, and report progress back via task state.

No tasks are registered yet — that starts in Part 4 (async job pipeline).
"""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "padel_ai_pipeline",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
