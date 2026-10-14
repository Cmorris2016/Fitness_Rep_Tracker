"""Count exercise repetitions from a webcam or video using MoveNet pose estimation.

Usage:
    python rep_counter.py --exercise curl
    python rep_counter.py --exercise squat --video path/to/clip.mp4
Press 'q' to quit, 'r' to reset the count.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).parent / "models" / "movenet_lightning.tflite"
INPUT_SIZE = 192  # MoveNet Lightning expects 192x192; Thunder expects 256x256

# MoveNet's 17 keypoints, in output order.
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}

SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# Each exercise tracks one joint angle (a-b-c, measured at b).
# A rep = angle goes above `extended`, then below `contracted`.
EXERCISES = {
    "curl":   {"joint": ("shoulder", "elbow", "wrist"), "extended": 150, "contracted": 50},
    "squat":  {"joint": ("hip", "knee", "ankle"),       "extended": 160, "contracted": 100},
    "pushup": {"joint": ("shoulder", "elbow", "wrist"), "extended": 150, "contracted": 90},
    "press":  {"joint": ("hip", "shoulder", "elbow"),   "extended": 150, "contracted": 80},
}

MIN_SCORE = 0.3  # ignore keypoints MoveNet isn't confident about


class MoveNet:
    def __init__(self, model_path=MODEL_PATH):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def detect(self, frame_bgr):
        """Return a (17, 3) array of (y, x, score), with y/x normalized to [0, 1]."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = tf.image.resize_with_pad(np.expand_dims(rgb, 0), INPUT_SIZE, INPUT_SIZE)
        self.interpreter.set_tensor(self.input["index"], tf.cast(img, self.input["dtype"]))
        self.interpreter.invoke()
        kps = self.interpreter.get_tensor(self.output["index"])[0, 0]
        return unpad(kps, frame_bgr.shape[:2])


def unpad(kps, frame_hw):
    """Map keypoints from the padded square input back to the original frame."""
    h, w = frame_hw
    side = max(h, w)
    kps = kps.copy()
    kps[:, 0] = (kps[:, 0] * side - (side - h) / 2) / h
    kps[:, 1] = (kps[:, 1] * side - (side - w) / 2) / w
    return kps


def joint_angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def tracked_angle(kps, joint, frame_hw):
    """Angle for the body side MoveNet sees most confidently, or None."""
    h, w = frame_hw
    best = None
    for side in ("left", "right"):
        idx = [KP[f"{side}_{part}"] for part in joint]
        score = kps[idx, 2].min()
        if score >= MIN_SCORE and (best is None or score > best[0]):
            # Convert to pixels so the angle isn't distorted by the frame's aspect ratio.
            pts = [kps[i, :2][::-1] * (w, h) for i in idx]
            best = (score, joint_angle(*pts), side)
    return (best[1], best[2]) if best else (None, None)


class RepCounter:
    def __init__(self, extended, contracted, smoothing=0.5):
        self.extended = extended
        self.contracted = contracted
        self.smoothing = smoothing
        self.reset()

    def reset(self):
        self.reps = 0
        self.stage = None
        self.angle = None

    def update(self, angle):
        if angle is None:
            return
        # Exponential moving average to damp keypoint jitter.
        if self.angle is None:
            self.angle = angle
        else:
            self.angle = self.smoothing * self.angle + (1 - self.smoothing) * angle

        if self.angle > self.extended:
            self.stage = "extended"
        elif self.angle < self.contracted and self.stage == "extended":
            self.stage = "contracted"
            self.reps += 1


def draw(frame, kps, counter, exercise, side):
    h, w = frame.shape[:2]
    for a, b in SKELETON:
        ka, kb = kps[KP[a]], kps[KP[b]]
        if ka[2] >= MIN_SCORE and kb[2] >= MIN_SCORE:
            cv2.line(frame, (int(ka[1] * w), int(ka[0] * h)),
                     (int(kb[1] * w), int(kb[0] * h)), (255, 200, 0), 2)
    for y, x, s in kps:
        if s >= MIN_SCORE:
            cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

    angle = "--" if counter.angle is None else f"{counter.angle:.0f}"
    cv2.rectangle(frame, (0, 0), (330, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"{exercise.upper()}  reps: {counter.reps}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"angle: {angle}  side: {side or '--'}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"stage: {counter.stage or '--'}", (10, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exercise", choices=EXERCISES, default="curl")
    parser.add_argument("--video", help="video file to read instead of the webcam")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    args = parser.parse_args()

    cfg = EXERCISES[args.exercise]
    model = MoveNet()
    counter = RepCounter(cfg["extended"], cfg["contracted"])

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open video source. On macOS, allow camera "
                         "access for your terminal in System Settings > Privacy & Security.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.video:
            frame = cv2.flip(frame, 1)  # mirror the webcam so it feels natural

        kps = model.detect(frame)
        angle, side = tracked_angle(kps, cfg["joint"], frame.shape[:2])
        counter.update(angle)
        draw(frame, kps, counter, args.exercise, side)

        cv2.imshow("MoveNet Rep Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            counter.reset()

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total {args.exercise} reps: {counter.reps}")


if __name__ == "__main__":
    main()
