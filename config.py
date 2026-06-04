from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Paths
    input_dir:   Path = Path("input")
    output_dir:  Path = Path("output")
    temp_dir:    Path = Path("temp")
    db_path:     Path = Path("data/processed.db")

    # Whisper
    whisper_model: str  = os.getenv("WHISPER_MODEL", "medium")
    use_gpu:       bool = os.getenv("USE_GPU", "false").lower() == "true"

    # Clip settings
    min_clip_duration: int = int(os.getenv("MIN_CLIP_SECONDS", "30"))
    max_clip_duration: int = int(os.getenv("MAX_CLIP_SECONDS", "90"))
    clips_per_video:   int = int(os.getenv("CLIPS_PER_VIDEO", "3"))
    output_resolution: str = "1080x1920"

    # Viral scoring weights (must sum to 1.0)
    audio_energy_weight: float = 0.35
    sentiment_weight:    float = 0.30
    llm_score_weight:    float = 0.35

    # Subtitle style
    subtitle_font:    str  = "Arial Bold"
    subtitle_size:    int  = 18
    subtitle_outline: bool = True

    # API keys
    openai_key:    str = os.getenv("OPENAI_API_KEY", "")
    use_local_llm: bool = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"


@dataclass
class JobContext:
    """Shared state object passed between all pipeline stages."""
    video_path:    Path = None
    audio_path:    Path = None
    transcript:    list = field(default_factory=list)    # [{start,end,text,words}]
    viral_moments: list = field(default_factory=list)    # [{start,end,score,text}]
    face_regions:  dict = field(default_factory=dict)    # {moment_idx: (x,y,w,h)}
    output_clips:  list = field(default_factory=list)    # [Path, ...]
    video_hash:    str  = ""