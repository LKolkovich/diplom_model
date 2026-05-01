from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.models import ModuleTag, TaskStatus


class SubmitResponse(BaseModel):
    task_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    stage: str
    message: str
    current_module: Optional[ModuleTag] = None
    error: Optional[str] = None


class ResultPendingResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int


class ResultErrorResponse(BaseModel):
    task_id: str
    status: TaskStatus
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    pipeline_busy: bool