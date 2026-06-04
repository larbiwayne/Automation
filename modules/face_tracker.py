import cv2
import mediapipe as mp
from loguru import logger

def track_faces(ctx, cfg):
    """
    For each viral moment, sample 1 frame every 2 seconds.
    Find average face center and compute 9:16 crop box around it.
    Falls back to center crop if no face detected.
    """
    cap = cv2.VideoCapture(str(ctx.video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_det = mp.solutions.face_detection.FaceDetection(
        model_selection=1,             # model 1 = better for full-body shots
        min_detection_confidence=0.5
    )

    # Desired output: 9:16. Compute crop width from video height.
    crop_w = int(H * 9 / 16)
    crop_w = min(crop_w, W)   # can't crop wider than source

    face_data = {}

    for idx, moment in enumerate(ctx.viral_moments):
        centers = []
        for t in range(int(moment["start"]), int(moment["end"]), 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if not ret: continue
            results = face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                b   = results.detections[0].location_data.relative_bounding_box
                cx  = (b.xmin + b.width / 2) * W
                centers.append(cx)

        avg_cx  = (sum(centers) / len(centers)) if centers else W / 2
        crop_x  = int(avg_cx - crop_w / 2)
        crop_x  = max(0, min(crop_x, W - crop_w))

        face_data[idx] = (crop_x, 0, crop_w, H)
        src = "face-centered" if centers else "center-fallback"
        logger.info(f"  Moment {idx+1} crop: x={crop_x} w={crop_w} [{src}]")

    cap.release()
    face_det.close()
    ctx.face_regions = face_data
    return ctx