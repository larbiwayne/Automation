from faster_whisper import WhisperModel
from loguru import logger

_model = None   # module-level cache — only loaded once per process

def _get_model(cfg):
    global _model
    if _model is None:
        device       = "cuda" if cfg.use_gpu else "cpu"
        compute_type = "float16" if cfg.use_gpu else "int8"
        logger.info(f"Loading Whisper '{cfg.whisper_model}' on {device} ({compute_type})")
        _model = WhisperModel(cfg.whisper_model, device=device, compute_type=compute_type)
    return _model


def transcribe(ctx, cfg):
    model = _get_model(cfg)

    segments_iter, info = model.transcribe(
        str(ctx.audio_path),
        beam_size=5,
        word_timestamps=True,   # REQUIRED for word-level subtitles
        language=None            # auto-detect language
    )

    logger.info(f"Detected language: {info.language} ({info.language_probability:.0%})")

    ctx.transcript = []
    for seg in segments_iter:   # iterator — streams segments as they complete
        ctx.transcript.append({
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text.strip(),
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "prob": w.probability}
                for w in (seg.words or [])
            ]
        })

    return ctx