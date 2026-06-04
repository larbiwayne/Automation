import subprocess, srt
from datetime import timedelta
from pathlib import Path
from loguru import logger

def _build_srt(ctx, moment_idx, moment):
    """Build word-level SRT with timestamps relative to clip start."""
    subs, counter = [], 1
    t0 = moment["start"]   # clip start time in source video

    for seg in ctx.transcript:
        if seg["end"] < moment["start"] or seg["start"] > moment["end"]:
            continue

        if seg.get("words"):
            # Group into 2-word chunks — TikTok-style karaoke captions
            words  = seg["words"]
            chunks = [words[i:i+2] for i in range(0, len(words), 2)]
            for chunk in chunks:
                cs   = max(0, chunk[0]["start"] - t0)
                ce   = chunk[-1]["end"] - t0
                text = " ".join(w["word"].strip().upper() for w in chunk)
                subs.append(srt.Subtitle(
                    index=counter,
                    start=timedelta(seconds=cs),
                    end=timedelta(seconds=max(cs + 0.3, ce)),
                    content=text
                ))
                counter += 1
        else:
            # Fallback: full segment as one subtitle line
            subs.append(srt.Subtitle(
                index=counter,
                start=timedelta(seconds=max(0, seg["start"] - t0)),
                end=timedelta(seconds=seg["end"] - t0),
                content=seg["text"].upper()
            ))
            counter += 1
    return srt.compose(subs)


def generate_subtitles(ctx, cfg):
    for idx, (moment, clip_path) in enumerate(zip(ctx.viral_moments, ctx.output_clips)):
        srt_path = clip_path.with_suffix(".srt")
        srt_text = _build_srt(ctx, idx, moment)
        srt_path.write_text(srt_text, encoding="utf-8")

        # Build subtitle style string for FFmpeg
        # PrimaryColour in BGR hex with alpha prefix: &HAABBGGRR
        style = (
            f"FontName={cfg.subtitle_font},"
            f"FontSize={cfg.subtitle_size},"
            "PrimaryColour=&H00FFFFFF,"   # white text
            "OutlineColour=&H00000000,"   # black outline
            "BackColour=&H80000000,"     # semi-transparent shadow
            "Outline=3,"
            "Shadow=1,"
            "Bold=1,"
            "Alignment=2,"               # bottom-center
            "MarginV=120"                # keep away from bottom edge
        )
        # Escape Windows backslashes for FFmpeg
        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
        vf  = f"subtitles='{srt_escaped}':force_style='{style}'"

        captioned = clip_path.with_stem(clip_path.stem + "_sub")
        cmd = [
            "ffmpeg", "-i", str(clip_path),
            "-vf", vf,
            "-c:a", "copy",   # don't re-encode audio
            str(captioned), "-y"
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            clip_path.unlink()
            captioned.rename(clip_path)
            srt_path.unlink()
            logger.info(f"  Subtitles burned: {clip_path.name}")
        else:
            logger.error(f"Subtitle burn failed: {result.stderr.decode()[-500:]}")
    return ctx