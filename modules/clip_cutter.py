import subprocess
from pathlib import Path
from loguru import logger

def cut_clips(ctx, cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_clips = []

    for idx, moment in enumerate(ctx.viral_moments):
        crop   = ctx.face_regions.get(idx)
        stem   = ctx.video_path.stem
        out    = cfg.output_dir / f"{stem}_clip{idx+1:02d}.mp4"

        # Build the FFmpeg video filter chain
        vf_parts = []
        if crop:
            x, y, w, h = crop
            vf_parts.append(f"crop={w}:{h}:{x}:{y}")

        # Scale to TikTok native resolution
        vf_parts.append("scale=1080:1920:flags=lanczos")
        # Sharpen slightly (optional but improves look after scale)
        vf_parts.append("unsharp=5:5:0.8:3:3:0.4")
        vf = ",".join(vf_parts)

        cmd = [
            "ffmpeg",
            "-ss", str(moment["start"]),   # start time
            "-to", str(moment["end"]),     # end time
            "-i",  str(ctx.video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",              # quality: 18=high, 28=low. 22 is good
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",  # web-friendly streaming
            str(out), "-y"
        ]

        logger.info(f"  Cutting clip {idx+1}: {moment['start']:.1f}s–{moment['end']:.1f}s")
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr.decode()[-500:]}")
            continue
        ctx.output_clips.append(out)
        logger.info(f"  Saved: {out.name}")

    return ctx