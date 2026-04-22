from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    hf_token: str = ""
    whisper_model: str = "large-v2"
    device: Literal["cuda", "cpu"] = "cpu"
    compute_type: Literal["float16", "int8_float16", "int8", "float32"] = "int8"
    mock_mode: bool = False

    frame_rate: int = 5
    output_dir: Path = Path("output")
    templates_dir: Path = Path("templates/agents")

    debug_frames: bool = False
    debug_frames_dir: Path = Path("debug_frames")

    voicechat_roi: str = "0.0,0.15,0.075,0.65"
    phash_threshold: int = 10

    @field_validator("output_dir", "templates_dir", "debug_frames_dir", mode="before")
    @classmethod
    def _ensure_path(cls, v: object) -> Path:
        p = Path(str(v))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def roi_as_tuple(self) -> Tuple[float, float, float, float]:
        parts = [float(x) for x in self.voicechat_roi.split(",")]
        if len(parts) != 4:
            raise ValueError("voicechat_roi must have exactly 4 comma-separated floats")
        return (parts[0], parts[1], parts[2], parts[3])


@lru_cache
def get_settings() -> Settings:
    return Settings()
