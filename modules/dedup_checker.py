import hashlib, sqlite3
from pathlib import Path
from loguru import logger

def _init_db(cfg):
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_videos (
            hash         TEXT PRIMARY KEY,
            filename     TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            clips_made   INTEGER DEFAULT 0
        )
    """)
    conn.commit(); conn.close()


def _file_hash(path):
    """Fast hash using first + last 2MB (avoids reading full large video files)."""
    h = hashlib.md5()
    size = path.stat().st_size
    with open(path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))
        if size > 4 * 1024 * 1024:
            f.seek(-2 * 1024 * 1024, 2)
            h.update(f.read(2 * 1024 * 1024))
    return h.hexdigest()


def is_duplicate(ctx, cfg) -> bool:
    _init_db(cfg)
    h = _file_hash(ctx.video_path)
    ctx.video_hash = h
    conn = sqlite3.connect(cfg.db_path)
    row  = conn.execute("SELECT 1 FROM processed_videos WHERE hash=?", (h,)).fetchone()
    conn.close()
    if row:
        logger.warning(f"Duplicate detected: {ctx.video_path.name} (hash={h[:8]}...)")
    return row is not None


def mark_processed(ctx, cfg):
    conn = sqlite3.connect(cfg.db_path)
    conn.execute(
        "INSERT OR IGNORE INTO processed_videos (hash, filename, clips_made) VALUES (?,?,?)",
        (ctx.video_hash, ctx.video_path.name, len(ctx.output_clips))
    )
    conn.commit(); conn.close()
    logger.info(f"Marked as processed in DB: {ctx.video_path.name}")