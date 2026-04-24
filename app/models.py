from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ModuleTag(str, Enum):
    m1_validate = "M1:AudioValidator"
    m2_frames = "M2:FrameExtractor"
    m3_asr = "M3:WhisperXASR"
    m4_cv = "M4:CVMicDetector"
    m5_fusion = "M5:FusionEngine"
    m6_export = "M6:SRTExporter"
    done = "done"


class TaskState(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.pending
    progress: int = Field(default=0, ge=0, le=100)
    stage: str = "INITIALIZING"
    message: str = "Waiting to start"
    current_module: Optional[ModuleTag] = None
    error: Optional[str] = None
    result_files: Dict[str, str] = Field(default_factory=dict)


class ProcessConfig(BaseModel):
    min_speakers: Optional[int] = Field(default=None, ge=1, le=10)
    max_speakers: Optional[int] = Field(default=None, ge=1, le=10)
    language: Optional[str] = Field(default=None)
    initial_prompt: Optional[str] = Field(default=None)
    video_mode: Optional[str] = Field(default=None)  # "full_frame" vs "roi_crop"


class ProcessResponse(BaseModel):
    task_id: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    stage: str
    message: str
    current_module: Optional[ModuleTag]
    error: Optional[str]


class ResultResponse(BaseModel):
    task_id: str
    status: TaskStatus
    files: Dict[str, str]


class SRTSegment(BaseModel):
    index: int
    start: float
    end: float
    speaker: str
    text: str


class FusedSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class ASRWord(BaseModel):
    word: str
    start: float
    end: float
    score: float = 0.0


class ASRSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"
    words: List[ASRWord] = Field(default_factory=list)


class CVDetection(BaseModel):
    timestamp: float
    agent_name: str
    confidence: float
