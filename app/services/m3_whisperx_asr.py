"""M3 – WhisperX ASR.

Runs transcription, forced alignment, and speaker diarization via the
WhisperX library.  Returns a list of ASRSegment objects whose *speaker*
field carries generic labels like "SPEAKER_00", "SPEAKER_01", … which are
later resolved to Valorant agent names by M5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.models import ASRSegment, ASRWord

logger = logging.getLogger(__name__)


def _import_whisperx():
    try:
        import whisperx  # type: ignore[import]
        return whisperx
    except ImportError as exc:
        raise ImportError(
            "whisperx is not installed. Run: pip install whisperx"
        ) from exc


def transcribe(
    audio_path: str | Path,
    model_name: str = "large-v2",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    hf_token: str = "",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    batch_size: int = 16,
    initial_prompt: Optional[str] = None,
) -> list[ASRSegment]:
    """Transcribe, align, and diarise *audio_path*.

    Parameters
    ----------
    audio_path:
        16 kHz mono WAV file produced by M1.
    model_name:
        Whisper model identifier (e.g. "large-v2").
    device:
        "cuda" or "cpu".
    compute_type:
        Quantisation level for faster-whisper backend.
    language:
        ISO-639-1 code. ``None`` triggers auto-detection.
    hf_token:
        HuggingFace token required by pyannote diarization models.
    min_speakers / max_speakers:
        Hint for the diarization model.
    batch_size:
        Whisper inference batch size.
    initial_prompt:
        Optional text to guide the model's style and vocabulary.

    Returns
    -------
    list[ASRSegment]
        Time-stamped segments with word-level detail and speaker labels.
    """
    wx = _import_whisperx()
    audio_path = Path(audio_path)

    logger.info("M3 – loading WhisperX model '%s' on %s (%s)", model_name, device, compute_type)
    model = wx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    logger.info("M3 – loading audio: %s", audio_path)
    audio = wx.load_audio(str(audio_path))

    logger.info("M3 – transcribing …")
    raw = model.transcribe(audio, batch_size=batch_size, initial_prompt=initial_prompt)
    detected_lang: str = raw.get("language", language or "en")
    logger.info("M3 – detected language: %s", detected_lang)

    logger.info("M3 – aligning …")
    align_model, align_meta = wx.load_align_model(
        language_code=detected_lang, device=device
    )
    aligned = wx.align(
        raw["segments"],
        align_model,
        align_meta,
        audio,
        device,
        return_char_alignments=False,
    )

    segments_with_speakers = aligned["segments"]

    if hf_token:
        logger.info("M3 – diarizing …")
        diarize_model = wx.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarize_kwargs: dict = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result_with_speakers = wx.assign_word_speakers(diarize_segments, aligned)
        segments_with_speakers = result_with_speakers["segments"]
    else:
        logger.warning(
            "M3 – HF_TOKEN not set; skipping pyannote diarization. "
            "All segments will be labelled 'SPEAKER_00'."
        )

    return _parse_segments(segments_with_speakers)


def _parse_segments(raw_segments: list) -> list[ASRSegment]:
    result: list[ASRSegment] = []
    for seg in raw_segments:
        words: list[ASRWord] = []
        for w in seg.get("words", []):
            words.append(
                ASRWord(
                    word=w.get("word", ""),
                    start=float(w.get("start", seg["start"])),
                    end=float(w.get("end", seg["end"])),
                    score=float(w.get("score", 0.0)),
                )
            )
        result.append(
            ASRSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
                speaker=seg.get("speaker", "SPEAKER_00"),
                words=words,
            )
        )
    return result
