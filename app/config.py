from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


    tmp_dir: Path = Path("tmp")
    auto_cleanup: bool = False
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_size_bytes: int = 2 * 1024 * 1024 * 1024


    # Где лежат шаблоны агентов
    templates_dir: Path = Field(default=Path("templates/agents"))

    # ROI голосового чата в долях (x1,y1,x2,y2)
    voicechat_roi: str = Field(default="0.0,0.45,0.075,0.75")

    # Discovery phase
    cv_discovery_fps: float = Field(default=1.0)
    cv_discovery_threshold: float = Field(default=0.75)
    cv_discovery_min_detections: int = Field(default=3)

    # Финальный проход
    cv_portrait_threshold: float = Field(default=0.70)
    cv_fallback_size_percent: float = Field(default=0.032)

    # Temporal smoothing K-of-N
    cv_temporal_k: int = Field(default=2)
    cv_temporal_n: int = Field(default=3)

    # Discovery / фильтрация агентов
    cv_skip_discovery: bool = Field(default=False)
    cv_agents_list: Optional[str] = Field(default=None)

    output_dir: Path = Field(default=Path("debug"))

    whisper_model: str = Field(
        default="large-v2",
        description="Whisper model identifier for WhisperX (e.g. large-v2, medium, small, large-v3)"
    )

    device: str = Field(default="cuda")
    compute_type: str = Field(default="float16")
    frame_rate: int = Field(default=5, description="Target FPS for frame extraction and CV detector")
# Hugging Face token для pyannote speaker diarization
    hf_token: str = Field(default="", description="HuggingFace API token for pyannote models")

    debug_frames: bool = True
    debug_frames_dir: Path = Path("debug/frames")
    # Debug для M4 (CV портретной детекции)
    debug_frames: bool = Field(
        default=False,
        description="Если True, сохранять debug-кадры с оверлеями из M4",
    )
    debug_dir: Path = Field(
        default=Path("debug/frames"),
        description="Куда сохранять debug-кадры M4",
    )
    debug_frames_dir: Path = Field(default=Path("debug/frames"))


    # Preprocessing
    cv_preprocess_mode: str = Field(
        default="none",
        description=(
            "Preprocessing mode for template matching: "
            "none, gray, gray_clahe, gray_blur, gray_clahe_blur, edges"
        ),
    )
    cv_preprocess_blur_kernel: int = Field(default=3)
    cv_preprocess_clahe_clip_limit: float = Field(default=2.0)
    cv_preprocess_clahe_tile_size: int = Field(default=8)
    cv_preprocess_canny_threshold1: int = Field(default=80)
    cv_preprocess_canny_threshold2: int = Field(default=160)

    # Геометрический кластеринг (множители от icon_size)
    cv_cluster_horizontal_tol: float = Field(
        default=0.25,
        description="Max horizontal distance между центрами (× icon_size)",
    )
    cv_cluster_vertical_min: float = Field(
        default=0.5,
        description="Min vertical distance между соседями (× icon_size)",
    )
    cv_cluster_vertical_max: float = Field(
        default=4.5,
        description="Max vertical distance между соседями (× icon_size)",
    )

    # Запас для Anchor Zone (множители от icon_size)
    cv_anchor_margin_x: float = Field(
        default=1.0,
        description="Horizontal margin для Anchor Zone (× icon_size)",
    )
    cv_anchor_margin_y: float = Field(
        default=1.5,
        description="Vertical margin для Anchor Zone (× icon_size)",
    )

    cv_cluster_mid_gap_bias: float = Field(
        default=0.35,
        description="Насколько близко gap_ratio должен быть к 1.5, чтобы считать его подозрительным",
    )

    # -------------------------------------------------------------------------
    # M5 Fusion — Phase 1 (hard matching)
    # -------------------------------------------------------------------------

    fusion_coverage_threshold: float = Field(
        default=0.60,
        description="Минимальный coverage для сильного speaker↔agent соответствия",
    )
    fusion_confidence_threshold: float = Field(
        default=0.65,
        description="Минимальная средняя confidence CV-интервалов для сильного соответствия",
    )
    fusion_hard_delta: float = Field(
        default=0.10,
        description="Минимальный отрыв лучшего agent coverage от второго кандидата",
    )
    fusion_forbidden_coverage_threshold: float = Field(
        default=0.05,
        description="Coverage ниже этого значения считается практически невозможной парой",
    )

    # -------------------------------------------------------------------------
    # M5 Fusion — Phase 2 (per-segment soft fusion)
    # -------------------------------------------------------------------------

    fusion_alpha: float = Field(
        default=0.5,
        description=(
            "Вес глобального diarization score (M3 global map + speaker_candidates). "
            "alpha + beta должны быть <= 1.0."
        ),
    )
    fusion_beta: float = Field(
        default=0.5,
        description=(
            "Вес per-segment CV overlap score (M4). "
            "alpha + beta должны быть <= 1.0."
        ),
    )
    fusion_winner_threshold: float = Field(
        default=0.15,
        description=(
            "Минимальный combined_score для принятия winner. "
            "Если ни один агент не достигает порога, сегмент остаётся unresolved."
        ),
    )
    fusion_hard_prior: float = Field(
        default=1.0,
        description=(
            "diar_score, который присваивается агенту из hard_map. "
            "1.0 = максимальный prior; агент из hard_map практически всегда выиграет, "
            "если только CV явно не противоречит."
        ),
    )

    def roi_as_tuple(self):
        parts = [float(x.strip()) for x in self.voicechat_roi.split(",")]
        if len(parts) != 4:
            raise ValueError("VOICECHAT_ROI must contain 4 comma-separated floats")
        return tuple(parts)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
