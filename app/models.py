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
    current_module: Optional[ModuleTag] = None
    error: Optional[str] = None
    result_files: Dict[str, str] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    video_path: str = Field(description="Absolute or relative path to the source video file")
    audio_path: Optional[str] = Field(
        default=None,
        description="Pre-extracted audio file path. When omitted the audio is extracted from video_path.",
    )
    language: Optional[str] = Field(
        default=None,
        description="ISO-639-1 language code, e.g. 'en'. Auto-detected when omitted.",
    )
    min_speakers: Optional[int] = Field(default=None, ge=1, le=10)
    max_speakers: Optional[int] = Field(default=None, ge=1, le=10)


class ProcessResponse(BaseModel):
    task_id: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
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
    frame_index: int
    timestamp: float
    agent: str
    confidence: float
