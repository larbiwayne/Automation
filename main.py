#!/usr/bin/env python
# main.py — TikTok Viral Clip Pipeline Orchestrator
# Usage: python main.py input\video.mp4
# Usage: python main.py input\video.mp4 --clips 5

from pathlib import Path
from loguru import logger
import subprocess, sys, shutil
import click

from config import Config, JobContext
from modules.transcriber    import transcribe
from modules.viral_detector  import detect_viral_moments
from modules.face_tracker    import track_faces
from modules.clip_cutter     import cut_clips
from modules.subtitle_gen    import generate_subtitles
from modules.dedup_checker   import is_duplicate, mark_processed

cfg = Config()

# Set up logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add("logs/pipeline_{time:YYYY-MM-DD}.log", rotation="1 day", retention="7 days")
Path("logs").mkdir(exist_ok=True)


def extract_audio(ctx: JobContext, cfg: Config) -> JobContext:
    """Extract mono 16kHz WAV from source video using FFmpeg."""
    out = cfg.temp_dir / "audio" / (ctx.video_path.stem + ".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", str(ctx.video_path),
        "-vn",              # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",    # 16kHz sample rate (Whisper standard)
        "-ac", "1",         # mono
        str(out), "-y"
    ], check=True, capture_output=True)
    ctx.audio_path = out
    logger.info(f"Audio extracted: {out}")
    return ctx


def cleanup_temp(ctx: JobContext, cfg: Config):
    """Delete temp files after successful processing."""
    try:
        if ctx.audio_path and ctx.audio_path.exists():
            ctx.audio_path.unlink()
        frame_dir = cfg.temp_dir / "frames"
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
    except Exception as e:
        logger.warning(f"Cleanup error (non-fatal): {e}")


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--clips", default=cfg.clips_per_video, show_default=True,
              help="Number of viral clips to extract")
@click.option("--skip-dedup", is_flag=True, default=False,
              help="Force reprocess even if already in database")
def run(video_path, clips, skip_dedup):
    """Process a video and output TikTok-ready viral clips."""
    cfg.clips_per_video = clips
    ctx = JobContext(video_path=Path(video_path))

    logger.info(f"=== Starting pipeline for: {ctx.video_path.name} ===")

    # Stage 0: Duplicate check
    if not skip_dedup and is_duplicate(ctx, cfg):
        logger.warning("Video already processed. Use --skip-dedup to force. Exiting.")
        return

    # Stage 1: Extract audio
    logger.info("[1/5] Extracting audio...")
    ctx = extract_audio(ctx, cfg)

    # Stage 2: Transcribe
    logger.info("[2/5] Transcribing with Whisper...")
    ctx = transcribe(ctx, cfg)
    logger.info(f"      {len(ctx.transcript)} segments transcribed")

    # Stage 3: Detect viral moments
    logger.info("[3/5] Scoring viral moments...")
    ctx = detect_viral_moments(ctx, cfg)

    # Stage 3b: Track faces (runs in parallel with scoring in production)
    logger.info("[3b]  Tracking faces for auto-crop...")
    ctx = track_faces(ctx, cfg)

    # Stage 4: Cut and reframe clips
    logger.info("[4/5] Cutting and encoding 9:16 clips...")
    ctx = cut_clips(ctx, cfg)

    # Stage 5: Burn subtitles
    logger.info("[5/5] Burning subtitles...")
    ctx = generate_subtitles(ctx, cfg)

    # Mark as processed and clean up
    mark_processed(ctx, cfg)
    cleanup_temp(ctx, cfg)

    logger.success(f"Done! {len(ctx.output_clips)} clips saved to {cfg.output_dir}/")
    for i, p in enumerate(ctx.output_clips):
        logger.success(f"  Clip {i+1}: {p.name}")


if __name__ == "__main__":
    Path("modules").mkdir(exist_ok=True)
    Path("modules/__init__.py").touch()
    for d in ["input", "output", "temp/audio", "temp/frames", "data", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    run()