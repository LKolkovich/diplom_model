from enum import Enum
from typing import Dict, List, Literal, Optional, Set

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
    video_mode: Optional[Literal["full_frame", "user_crop"]] = Field(default=None)


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


class SpeakerAgentCandidate(BaseModel):
    """Per-segment attribution candidate produced by M5 Phase 2."""

    agent: str
    score: float        # combined score: alpha*diar_score + beta*cv_score
    diar_score: float   # contribution from M3 global map + speaker_candidates
    cv_score: float     # contribution from M4 per-segment overlap


class FusedSegment(BaseModel):
    start: float
    end: float
    speaker: str        # итоговый агент/лейбл (или SPEAKER_XX если unresolved)
    text: str
    speaker_candidates: List[SpeakerAgentCandidate] = Field(default_factory=list)
    attribution_confidence: float = 0.0
    attribution_source: str = "unresolved"
    # "cv+diar" | "cv_only" | "map_only" | "unresolved"


class ASRWord(BaseModel):
    word: str
    start: float
    end: float
    score: float = 0.0


class ASRSpeakerCandidate(BaseModel):
    speaker_id: str
    weight: float  # 0..1, не обязательно нормировано


class ASRSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"
    words: List[ASRWord] = Field(default_factory=list)
    speaker_candidates: List[ASRSpeakerCandidate] = Field(default_factory=list)


class CVDetection(BaseModel):
    timestamp: float
    agent_name: str
    confidence: float


class DiscoveryResult(BaseModel):
    active_agents: Set[str]
    median_scale: float
    anchor_zone: Optional[tuple[int, int, int, int]]
    stats: Dict[str, Dict]  # {agent_name: {count, scales, bboxes}}
