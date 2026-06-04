import numpy as np
import librosa
from transformers import pipeline
from loguru import logger

_sentiment_pipe = None

def _get_sentiment():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        # Runs 100% locally — no API needed. Downloads ~500MB on first run.
        logger.info("Loading sentiment model (first run downloads ~500MB)...")
        _sentiment_pipe = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None
        )
    return _sentiment_pipe


def score_audio_energy(audio_path, start, end):
    """
    Returns 0.0–1.0 based on RMS audio energy in the time window.
    High energy = loud = reactions, applause, intense speech.
    """
    y, sr = librosa.load(str(audio_path), sr=16000, offset=start, duration=end-start)
    if len(y) == 0: return 0.0
    rms     = librosa.feature.rms(y=y)[0]
    # Log scale normalization (matches human hearing perception)
    score   = float(np.mean(np.log1p(rms) / np.log1p(0.5)))
    return min(1.0, max(0.0, score))


def score_sentiment(text):
    """
    Returns 0.0–1.0 emotional intensity score.
    Key insight: BOTH high-positive and high-negative go viral.
    Neutral content scores low.
    """
    if not text.strip(): return 0.0
    pipe    = _get_sentiment()
    results = pipe(text[:512])[0]
    pos = next((r["score"] for r in results if r["label"] == "positive"), 0)
    neg = next((r["score"] for r in results if r["label"] == "negative"), 0)
    return max(pos, neg)


def score_with_gpt(text, cfg):
    """GPT-4o-mini scoring (~$0.001 per clip). Falls back to 0.5 if no key."""
    if not cfg.openai_key: return 0.5
    from openai import OpenAI
    client = OpenAI(api_key=cfg.openai_key)
    prompt = f"""You are a TikTok content strategist. Rate this transcript segment's
viral potential from 0.0 to 1.0. Consider: strong hooks, emotional moments,
surprising facts, humor, relatable situations, conflict/tension.
Reply with ONLY a decimal number between 0.0 and 1.0.

TRANSCRIPT:
{text[:800]}"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )
        return float(resp.choices[0].message.content.strip())
    except:
        return 0.5


def score_with_ollama(text):
    """Free local LLM scoring via Ollama. Requires 'ollama serve' running."""
    import requests
    prompt = (f"Rate this clip 0.0-1.0 for TikTok virality. Only reply with a decimal number."
              f"\n\nTEXT: {text[:600]}")
    try:
        r = requests.post("http://localhost:11434/api/generate",
            json={"model": "llama3.1:8b", "prompt": prompt, "stream": False},
            timeout=60)
        return float(r.json()["response"].strip())
    except:
        return 0.5


def detect_viral_moments(ctx, cfg):
    """
    Sliding window analysis over full transcript.
    Scores each window with 3 signals, then selects top N non-overlapping clips.
    """
    if not ctx.transcript:
        logger.warning("No transcript found — cannot score moments")
        return ctx

    total_dur = ctx.transcript[-1]["end"]
    window    = cfg.min_clip_duration
    step      = window // 2   # 50% overlap between windows
    windows   = []

    logger.info(f"Scanning {total_dur:.0f}s video with {window}s windows...")

    for t_start in range(0, int(total_dur - window), step):
        t_end   = min(t_start + window, int(total_dur))
        segs    = [s for s in ctx.transcript
                   if s["start"] >= t_start and s["end"] <= t_end]
        text    = " ".join(s["text"] for s in segs)

        audio_s = score_audio_energy(ctx.audio_path, t_start, t_end)
        senti_s = score_sentiment(text)

        if cfg.use_local_llm:
            llm_s = score_with_ollama(text)
        else:
            llm_s = score_with_gpt(text, cfg)

        score = (
            cfg.audio_energy_weight * audio_s +
            cfg.sentiment_weight    * senti_s +
            cfg.llm_score_weight    * llm_s
        )
        windows.append({"start": t_start, "end": t_end,
                        "score": score, "text": text})

    # Greedy non-overlapping selection
    windows.sort(key=lambda x: x["score"], reverse=True)
    selected, used = [], []
    for w in windows:
        overlap = any(not (w["end"] <= u["start"] or w["start"] >= u["end"])
                      for u in used)
        if not overlap:
            selected.append(w); used.append(w)
        if len(selected) >= cfg.clips_per_video: break

    ctx.viral_moments = selected
    for i, m in enumerate(selected):
        logger.info(f"  Moment {i+1}: {m['start']:.0f}s–{m['end']:.0f}s | score={m['score']:.3f}")
    return ctx