"""M3 – WhisperX ASR.

Runs transcription, forced alignment, and speaker diarization via the
WhisperX library. Returns a list of ASRSegment objects whose *speaker*
field carries generic labels like "SPEAKER_00", "SPEAKER_01", … which are
later resolved to Valorant agent names by M5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.models import ASRSegment, ASRWord
from app.models import ASRSegment, ASRWord, ASRSpeakerCandidate

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
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = "ru",
        hf_token: str = "",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        batch_size: int = 16,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        vad_onset: float = 0.500,
        vad_offset: float = 0.363,
        chunk_size: int = 30,
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
        ISO-639-1 code. Defaults to "ru". Pass ``None`` to enable auto-detection.
    hf_token:
        HuggingFace token required by pyannote diarization models.
    min_speakers / max_speakers:
        Hints for the diarization model.
    batch_size:
        Whisper inference batch size.
    initial_prompt:
        Optional prompt with domain vocabulary (e.g. agent names, in-game slang).
    vad_filter:
        Kept for API compatibility; VAD itself is configured through vad_options.
    vad_onset / vad_offset:
        VAD thresholds passed into WhisperX load_model(..., vad_options=...).
    chunk_size:
        VAD chunk size in seconds.

    Returns
    -------
    list[ASRSegment]
        Time-stamped segments with word-level detail and speaker labels.
    """
    wx = _import_whisperx()
    audio_path = Path(audio_path)

    asr_options: dict = {}
    if initial_prompt:
        asr_options["initial_prompt"] = initial_prompt

    vad_options: dict | None = None
    if vad_filter:
        vad_options = {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
            "chunk_size": chunk_size,
        }

    logger.info(
        "M3 – loading WhisperX model '%s' on %s (%s), language=%s, "
        "vad_filter=%s, vad_onset=%.3f, vad_offset=%.3f, chunk_size=%d, prompt=%s",
        model_name,
        device,
        compute_type,
        language or "auto",
        vad_filter,
        vad_onset,
        vad_offset,
        chunk_size,
        "yes" if initial_prompt else "no",
        )

    load_model_kwargs: dict = {
        "device": device,
        "compute_type": compute_type,
        "language": language,
    }
    if asr_options:
        load_model_kwargs["asr_options"] = asr_options
    if vad_options:
        load_model_kwargs["vad_options"] = vad_options

    model = wx.load_model(
        model_name,
        **load_model_kwargs,
    )

    logger.info("M3 – loading audio: %s", audio_path)
    audio = wx.load_audio(str(audio_path))

    logger.info("M3 – transcribing … (batch_size=%d)", batch_size)
    raw = model.transcribe(audio, batch_size=batch_size)

    detected_lang: str = raw.get("language", language or "ru")
    logger.info("M3 – detected/used language: %s", detected_lang)

    logger.info("M3 – aligning …")
    align_model, align_meta = wx.load_align_model(
        language_code=detected_lang,
        device=device,
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
        diarize_model = wx.diarize.DiarizationPipeline(
            token=hf_token,
            device=device,
        )

        diarize_kwargs: dict = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

        if diarize_kwargs:
            logger.info(
                "M3 – diarization speaker hint: min=%s, max=%s",
                min_speakers,
                max_speakers,
            )

        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result_with_speakers = wx.assign_word_speakers(diarize_segments, aligned)
        segments_with_speakers = result_with_speakers["segments"]

        speaker_set = {
            seg.get("speaker", "SPEAKER_00")
            for seg in segments_with_speakers
        }
        total_duration = sum(
            float(seg["end"]) - float(seg["start"])
            for seg in segments_with_speakers
        )
        logger.info(
            "M3 – diarization complete: %d speaker(s) detected %s, total speech duration: %.1fs",
            len(speaker_set),
            sorted(speaker_set),
            total_duration,
        )
    else:
        logger.warning(
            "M3 – HF_TOKEN not set; skipping pyannote diarization. "
            "All segments will be labelled 'SPEAKER_00'."
        )

    return _parse_segments(segments_with_speakers)

def _parse_segments(raw_segments: list) -> list[ASRSegment]:
    """Convert WhisperX raw segments into ASRSegment objects.

    Эвристически добавляет speaker_candidates на основе времной близости
    и перекрытия других спикеров.
    """
    result: list[ASRSegment] = []

    # Сначала соберём "простую" структуру для удобства:
    # список (index, speaker_id, start, end) для всех сегментов.
    meta: list[tuple[int, str, float, float]] = []
    for idx, seg in enumerate(raw_segments):
        spk = seg.get("speaker", "SPEAKER_00")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        meta.append((idx, spk, start, end))

    # Предварительно построим список кандидатов по временной близости
    # для каждого сегмента: {idx -> {speaker_id -> weight}}
    candidates_by_idx: dict[int, dict[str, float]] = {}

    for i, (idx_i, spk_i, start_i, end_i) in enumerate(meta):
        duration_i = max(0.0, end_i - start_i)
        if duration_i <= 0:
            continue

        local: dict[str, float] = {}

        for j, (idx_j, spk_j, start_j, end_j) in enumerate(meta):
            if i == j:
                continue
            if spk_j == spk_i:
                # тот же спикер — альтернативой не считаем, только primary
                continue

            # Перекрытие по времени
            overlap = max(0.0, min(end_i, end_j) - max(start_i, start_j))

            if overlap > 0:
                # Если есть реальное перекрытие, используем долю перекрытия
                # как вес (ограничиваем 0..1).
                w = min(1.0, overlap / duration_i)
            else:
                # Если нет перекрытия, но сегменты близко по времени,
                # используем затухающую функцию 1 / (1 + dist),
                # где dist — расстояние между сегментами в секундах.
                if end_j <= start_i:
                    dist = start_i - end_j
                else:
                    dist = start_j - end_i
                # Игнорируем слишком далёкие (например > 3 секунд)
                if dist > 3.0:
                    continue
                w = 1.0 / (1.0 + dist)

            # Агрегируем максимум по каждому speaker_id
            prev = local.get(spk_j, 0.0)
            if w > prev:
                local[spk_j] = w

        candidates_by_idx[idx_i] = local

    # Теперь собираем итоговые ASRSegment с words и speaker_candidates.
    total_segments = len(raw_segments)
    segments_with_alts = 0
    total_candidates_count = 0

    for idx, seg in enumerate(raw_segments):
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

        speaker_id = seg.get("speaker", "SPEAKER_00")
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg.get("text", "").strip()

        # Формируем список кандидатов:
        # 1) основной спикер с weight=1.0
        # 2) top-K (до 3) альтернативных спикеров из candidates_by_idx[idx]
        raw_candidates = candidates_by_idx.get(idx, {})
        # сортируем по убыванию веса
        alt_sorted = sorted(
            raw_candidates.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

        # ограничиваем количество альтернатив, чтобы не раздувать структуру
        TOP_K = 3
        alt_sorted = alt_sorted[:TOP_K]

        speaker_candidates: list[ASRSpeakerCandidate] = []

        # основной спикер всегда присутствует как первый кандидат
        speaker_candidates.append(
            ASRSpeakerCandidate(speaker_id=speaker_id, weight=1.0)
        )

        for spk_alt, w in alt_sorted:
            speaker_candidates.append(
                ASRSpeakerCandidate(speaker_id=spk_alt, weight=float(w))
            )

        if len(speaker_candidates) > 1:
            segments_with_alts += 1
            total_candidates_count += len(speaker_candidates)

        result.append(
            ASRSegment(
                start=start,
                end=end,
                text=text,
                speaker=speaker_id,
                words=words,
                speaker_candidates=speaker_candidates,
            )
        )

    # Логирование статистики по кандидатам
    if segments_with_alts > 0:
        avg_candidates = total_candidates_count / segments_with_alts
    else:
        avg_candidates = 0.0

    logger.info(
        "M3 – speaker candidates: %d/%d segments with alternatives, "
        "avg candidates (including primary) among them: %.2f",
        segments_with_alts,
        total_segments,
        avg_candidates,
    )

    # Для отладки выведем парочку примеров
    for seg in result[:2]:
        if seg.speaker_candidates:
            logger.debug(
                "M3 – example segment %.2f–%.2f speaker=%s candidates=%s",
                seg.start,
                seg.end,
                seg.speaker,
                [(c.speaker_id, round(c.weight, 3)) for c in seg.speaker_candidates],
            )

    return result
