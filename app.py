import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from typing import Generator

from flask import Flask, Response, render_template, request, send_file
import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

# Reduce CPU oversubscription and memory spikes (helpful on small hosts like Render).
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("NUMEXPR_NUM_THREADS", "1"))

import onnxruntime as ort

# NOTE: torch and ultralytics are NOT imported at module level.
# They consume ~200-300 MB just on import, which causes OOM on Render's
# 512 MB free tier.  They are imported lazily inside _load_one_model()
# only when a .pt model is actually loaded (the ONNX path never needs them).

app = Flask(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))

_model_lock = threading.Lock()
_predict_lock = threading.Lock()

# Each entry is a dict:
# {"name": str, "path": str, "model": object, "type": "yolov8"|"yolov5"|"onnxrt"}
_models = None


def _default_model_specs() -> list[dict]:
    """Return the default 3-model ensemble spec.

    User requirement: always run all three models by default.
    """
    prefer_onnx = os.environ.get("PREFER_ONNX", "1").strip().lower() not in {"0", "false", "no"}

    # Render/Linux hosts often have tight memory; ONNX Runtime is much lighter than loading 3x PyTorch models.
    if prefer_onnx and os.name != "nt":
        onnx_specs = [
            {"name": "best", "path": os.path.join(ROOT, "model", "best.onnx")},
            {"name": "best_old", "path": os.path.join(ROOT, "model", "best_old.onnx")},
            {"name": "yolo11n", "path": os.path.join(ROOT, "yolo11n.onnx")},
        ]
        if all(os.path.exists(s["path"]) for s in onnx_specs):
            return onnx_specs

    return [
        {"name": "best", "path": os.path.join(ROOT, "model", "best.pt")},
        {"name": "best_old", "path": os.path.join(ROOT, "model", "best_old.pt")},
        {"name": "yolo11n", "path": os.path.join(ROOT, "yolo11n.pt")},
    ]


# ---------------------------------------------------------------------------
#  Direct ONNX Runtime helpers (bypass Ultralytics AutoBackend entirely)
# ---------------------------------------------------------------------------

def _letterbox(img: np.ndarray, new_shape: int = 640,
               color: tuple[int, int, int] = (114, 114, 114)):
    """Resize + pad to *new_shape* square, preserving aspect ratio.

    Returns (padded_img, scale_ratio, (pad_w, pad_h)).
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape - new_w) / 2, (new_shape - new_h) / 2

    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray,
               iou_threshold: float = 0.45) -> list[int]:
    """Pure-numpy greedy NMS.  *boxes* is (N, 4) as x1y1x2y2."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def _predict_boxes_onnxrt(session: ort.InferenceSession, frame: np.ndarray,
                          conf_thres: float, imgsz: int,
                          device: str | None = None) -> list[tuple]:
    """Run YOLO-style detection via ONNX Runtime directly.

    Works with both YOLOv8/v11 ONNX exports (output shape [1, 4+nc, N])
    and YOLOv5-style exports ([1, N, 4+nc+1]).
    """
    orig_h, orig_w = frame.shape[:2]

    # --- pre-process --------------------------------------------------------
    img, ratio, (dw, dh) = _letterbox(frame, imgsz)
    img = img[:, :, ::-1]                           # BGR → RGB
    img = img.transpose(2, 0, 1)                    # HWC → CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = img[np.newaxis, ...]                      # add batch dim

    # --- run ----------------------------------------------------------------
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})[0]  # first output tensor

    if output.ndim != 3:
        return []

    # --- normalise shape to (N, 4+nc) --------------------------------------
    # YOLOv8/v11: [1, 4+nc, N]  →  transpose   (shape[1] < shape[2])
    # YOLOv5:     [1, N, 5+nc]  →  already correct orientation
    is_v8_layout = output.shape[1] < output.shape[2]
    if is_v8_layout:
        preds = output[0].T                         # (N, 4+nc)
    else:
        preds = output[0]                           # (N, 5+nc) — YOLOv5

    num_cols = preds.shape[1]

    # YOLOv5 has an objectness column at index 4; YOLOv8+ does not.
    # We distinguish by the *original* tensor layout, not just column count,
    # because a YOLOv8 model with many classes can also have num_cols >= 6.
    if not is_v8_layout and num_cols >= 6:           # YOLOv5 (4+1+nc)
        obj = preds[:, 4]
        cls_scores = preds[:, 5:] * obj[:, None]
    else:                                           # YOLOv8/v11 (4+nc)
        cls_scores = preds[:, 4:]

    max_scores = cls_scores.max(axis=1)
    mask = max_scores >= conf_thres
    preds, max_scores = preds[mask], max_scores[mask]
    if len(preds) == 0:
        return []

    # cx, cy, w, h → x1, y1, x2, y2
    boxes = np.empty((len(preds), 4), dtype=np.float32)
    boxes[:, 0] = preds[:, 0] - preds[:, 2] / 2
    boxes[:, 1] = preds[:, 1] - preds[:, 3] / 2
    boxes[:, 2] = preds[:, 0] + preds[:, 2] / 2
    boxes[:, 3] = preds[:, 1] + preds[:, 3] / 2

    # NMS
    keep = _nms_numpy(boxes, max_scores, iou_threshold=0.45)

    out: list[tuple] = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        # undo letterbox padding + scale
        x1 = float(np.clip((x1 - dw) / ratio, 0, orig_w))
        y1 = float(np.clip((y1 - dh) / ratio, 0, orig_h))
        x2 = float(np.clip((x2 - dw) / ratio, 0, orig_w))
        y2 = float(np.clip((y2 - dh) / ratio, 0, orig_h))
        out.append((x1, y1, x2, y2, float(max_scores[i])))
    return out


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def _load_one_model(model_path: str):
    """Return (model, model_type) where model_type is 'yolov8', 'yolov5', or 'onnxrt'."""
    app.logger.info("Loading model from: %s", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()

    # ----- ONNX: load with onnxruntime directly (avoids Ultralytics
    #       AutoBackend metadata issues such as KeyError: 'task'). -----
    if ext == ".onnx":
        providers = ["CPUExecutionProvider"]
        try:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass
        session = ort.InferenceSession(model_path, providers=providers)
        inp = session.get_inputs()[0]
        app.logger.info(
            "ONNX model loaded via onnxruntime  input=%s  shape=%s  providers=%s",
            inp.name, inp.shape, session.get_providers(),
        )
        return session, "onnxrt"

    # ----- .pt / other: try Ultralytics first, fall back to YOLOv5 -----
    #  Lazy-import torch + ultralytics to avoid ~200-300 MB overhead when
    #  only ONNX models are used (e.g. Render free tier).
    from ultralytics import YOLO  # noqa: F811  (intentionally lazy)

    try:
        model = YOLO(model_path)
        model_type = "yolov8"
        app.logger.info("Model loaded successfully as YOLOv8/YOLO11")
        return model, model_type
    except Exception as e:
        # Common case: a YOLOv5-trained .pt is not forwards-compatible with Ultralytics YOLOv8/YOLO11.
        msg = str(e) or ""
        app.logger.warning("YOLO() failed: %s", msg)

        yolov5_incompatible = isinstance(e, TypeError) and "YOLOv5 model" in msg
        missing_yolov5_code = isinstance(e, ModuleNotFoundError) and "No module named 'models'" in msg

        if yolov5_incompatible or missing_yolov5_code:
            app.logger.warning("%s looks like YOLOv5 format. Loading from local yolov5_local...", model_path)
            try:
                yolov5_dir = os.path.join(ROOT, "yolov5_local")
                if yolov5_dir not in sys.path:
                    sys.path.insert(0, yolov5_dir)

                from hubconf import custom

                model = custom(path=model_path)
                model_type = "yolov5"
                app.logger.info("Model loaded successfully as YOLOv5 from local code")
                if hasattr(model, "names"):
                    model.names = {k: "defect" for k in model.names}
                return model, model_type
            except Exception as e2:
                app.logger.exception("Local YOLOv5 load failed: %s", e2)
                raise RuntimeError(f"Failed to load YOLOv5 model from '{model_path}': {e2}") from e
        raise


def get_models() -> list[dict]:
    """Return list of loaded models.

    By default, loads and uses all three: best.pt, best(old).pt, yolo11n.pt.

    Optional override (no UI prompt): set MODEL_PATHS to a semicolon-separated list.
    Example: MODEL_PATHS='model/best.pt;model/best(old).pt;yolo11n.pt'
    """
    global _models
    if _models is not None:
        return _models
    with _model_lock:
        if _models is None:
            specs = []

            model_paths_env = os.environ.get("MODEL_PATHS")
            if model_paths_env:
                parts = [p.strip() for p in model_paths_env.split(";") if p.strip()]
                for idx, p in enumerate(parts):
                    p_abs = p if os.path.isabs(p) else os.path.join(ROOT, p)
                    specs.append({"name": f"model{idx+1}", "path": p_abs})
            else:
                specs = _default_model_specs()

            # Load models sequentially to keep peak memory low
            # (important on Render 512 MB free tier).
            loaded = []
            for spec in specs:
                app.logger.info("Loading model '%s' from %s ...", spec["name"], spec["path"])
                model, model_type = _load_one_model(spec["path"])
                loaded.append({"name": spec["name"], "path": spec["path"], "model": model, "type": model_type})
                app.logger.info("  -> '%s' loaded as %s", spec["name"], model_type)

            _models = loaded
        return _models

_live_lock = threading.Lock()
_live_state = {"defect": False, "score": 0.0, "ts": 0.0}


def _set_live_state(defect: bool, score: float) -> None:
    with _live_lock:
        _live_state["defect"] = bool(defect)
        _live_state["score"] = float(score)
        _live_state["ts"] = time.time()


def _get_live_state() -> dict:
    with _live_lock:
        return dict(_live_state)


def _ensure_uint8_bgr(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _draw_box(img: np.ndarray, xyxy: tuple[float, float, float, float], label: str) -> None:
    x1, y1, x2, y2 = xyxy
    x1i, y1i = int(max(0, round(x1))), int(max(0, round(y1)))
    x2i, y2i = int(max(0, round(x2))), int(max(0, round(y2)))
    cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
    if label:
        cv2.putText(img, label, (x1i, max(0, y1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def _predict_boxes_yolov5(model, frame: np.ndarray, conf_thres: float, imgsz: int, device: str | None):
    model.conf = conf_thres
    if device:
        model.to(device)
    results = model(frame, size=imgsz)
    dets = results.xyxy[0]
    if dets is None or len(dets) == 0:
        return []
    out = []
    for row in dets:
        x1, y1, x2, y2, conf, cls = row.tolist()
        out.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
    return out


def _predict_boxes_yolov8(model, frame: np.ndarray, conf_thres: float, imgsz: int, device: str | None):
    results = model.predict(
        source=frame,
        conf=conf_thres,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy
    conf = boxes.conf
    out = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        c = float(conf[i].item()) if conf is not None else 0.0
        out.append((float(x1), float(y1), float(x2), float(y2), c))
    return out


def predict_and_annotate(frame: np.ndarray, conf_thres: float) -> tuple[np.ndarray, bool, float]:
    """Run all 3 models and draw the union of detections."""
    imgsz = int(os.environ.get("MODEL_IMGSZ", "640"))
    device = os.environ.get("MODEL_DEVICE")  # e.g. 'cpu' or '0'

    models = get_models()
    boxed = _ensure_uint8_bgr(frame.copy())

    any_defect = False
    max_conf = 0.0

    with _predict_lock:
        for idx, entry in enumerate(models):
            m = entry["model"]
            mtype = entry["type"]
            mname = entry["name"]

            if mtype == "onnxrt":
                dets = _predict_boxes_onnxrt(m, frame, conf_thres, imgsz, device)
            elif mtype == "yolov5":
                dets = _predict_boxes_yolov5(m, frame, conf_thres, imgsz, device)
            else:
                dets = _predict_boxes_yolov8(m, frame, conf_thres, imgsz, device)

            for x1, y1, x2, y2, c in dets:
                any_defect = True
                max_conf = max(max_conf, float(c))
                _draw_box(boxed, (x1, y1, x2, y2), f"defect {c:.2f}")

            if dets:
                cv2.putText(
                    boxed,
                    f"{mname}: {len(dets)}",
                    (10, 24 + 22 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

    return boxed, any_defect, max_conf


@app.route("/")
def index():
    # Render/Linux servers don't have a physical webcam device, but we can still support "Live"
    # by using the *client* (browser) webcam and POSTing frames to the server.
    enable_live = os.environ.get("ENABLE_LIVE", "1").strip().lower() not in {"0", "false", "no"}
    live_mode = os.environ.get("LIVE_MODE") or ("server" if os.name == "nt" else "client")
    live_mode = "server" if str(live_mode).strip().lower() == "server" else "client"
    # Client-live tuning knobs (mostly for Render/free CPU boxes).
    live_client_interval_ms = int(os.environ.get("LIVE_CLIENT_INTERVAL_MS", "700"))
    live_client_width = int(os.environ.get("LIVE_CLIENT_WIDTH", "480"))
    live_client_jpeg_quality = float(os.environ.get("LIVE_CLIENT_JPEG_QUALITY", "0.75"))

    return render_template(
        "index.html",
        enable_live=enable_live,
        live_mode=live_mode,
        live_client_interval_ms=live_client_interval_ms,
        live_client_width=live_client_width,
        live_client_jpeg_quality=live_client_jpeg_quality,
    )


@app.get("/live-status")
def live_status():
    s = _get_live_state()
    # Consider the stream "active" if it has updated recently.
    age = time.time() - float(s.get("ts") or 0.0)
    return {
        "defect": bool(s.get("defect")),
        "score": float(s.get("score") or 0.0),
        "age": float(age),
        "active": age < 2.0,
    }


@app.post("/upload-image")
def upload_image():
    f = request.files.get("file")
    if not f:
        return {"error": "No file"}, 400

    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image"}, 400

    conf_thres = float(os.environ.get("CONF_THRES", "0.1"))
    try:
        boxed, defect, score = predict_and_annotate(img, conf_thres)
    except Exception as e:
        app.logger.exception("Inference failed for /upload-image")
        msg = (str(e) or "").strip()
        if len(msg) > 600:
            msg = msg[:600] + "…"
        return {"error": "Inference failed", "type": type(e).__name__, "message": msg}, 500

    ok, enc = cv2.imencode(".jpg", boxed)
    if not ok:
        return {"error": "Encode failed"}, 500

    resp = send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg")
    resp.headers["X-Defect-Detected"] = "1" if defect else "0"
    resp.headers["X-Heatmap-Max"] = f"{score:.6f}"
    return resp


@app.post("/detect-image")
def detect_image_legacy():
    return upload_image()


@app.post("/upload-video")
def upload_video():
    f = request.files.get("file")
    if not f:
        return {"error": "No file"}, 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as inp:
        inp.write(f.read())
        in_path = inp.name

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        os.unlink(in_path)
        return {"error": "Could not open video"}, 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    conf_thres = float(os.environ.get("CONF_THRES", "0.25"))

    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)
    frame_dir = tempfile.mkdtemp(prefix="frames_")

    max_score, defect_any = 0.0, False
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            boxed, defect, score = predict_and_annotate(frame, conf_thres)
        except Exception as e:
            cap.release()
            os.unlink(in_path)
            shutil.rmtree(frame_dir, ignore_errors=True)
            os.unlink(out_path)
            app.logger.exception("Inference failed for /upload-video")
            msg = (str(e) or "").strip()
            if len(msg) > 600:
                msg = msg[:600] + "…"
            return {"error": "Inference failed", "type": type(e).__name__, "message": msg}, 500
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
        ok = cv2.imwrite(frame_path, boxed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            cap.release()
            os.unlink(in_path)
            shutil.rmtree(frame_dir, ignore_errors=True)
            os.unlink(out_path)
            return {"error": "Could not encode frames"}, 500

        frame_idx += 1
        max_score = max(max_score, score)
        defect_any |= defect

    cap.release()
    os.unlink(in_path)

    if frame_idx == 0:
        shutil.rmtree(frame_dir, ignore_errors=True)
        os.unlink(out_path)
        return {"error": "Empty video"}, 400

    # Encode frames to browser-friendly H.264 MP4.
    try:
        ffmpeg = get_ffmpeg_exe()
        input_pattern = os.path.join(frame_dir, "frame_%06d.jpg")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-start_number",
            "0",
            "-framerate",
            str(float(fps)),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            err = err[:800] if err else "ffmpeg failed"
            shutil.rmtree(frame_dir, ignore_errors=True)
            os.unlink(out_path)
            return {"error": f"Video encoding failed: {err}"}, 500

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            shutil.rmtree(frame_dir, ignore_errors=True)
            if os.path.exists(out_path):
                os.unlink(out_path)
            return {"error": "Video encoding failed: empty output"}, 500

    except Exception as e:
        shutil.rmtree(frame_dir, ignore_errors=True)
        if os.path.exists(out_path):
            os.unlink(out_path)
        return {"error": f"Video encoding failed: {e}"}, 500
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    resp = send_file(out_path, mimetype="video/mp4")
    resp.headers["X-Defect-Detected"] = "1" if defect_any else "0"
    resp.headers["X-Heatmap-Max"] = f"{max_score:.6f}"

    @resp.call_on_close
    def cleanup():
        os.unlink(out_path)

    return resp


@app.post("/detect-video")
def detect_video_legacy():
    return upload_video()


@app.route("/live")
def live():
    if os.environ.get("ENABLE_LIVE", "1").strip().lower() in {"0", "false", "no"}:
        return Response(
            "Live camera is disabled on this deployment.",
            status=503,
            mimetype="text/plain",
        )
    # Try to open the camera up-front so we can return a clear error.
    if os.name == "nt":
        test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        test_cap.release()
        return Response("Camera not available", status=500, mimetype="text/plain")
    test_cap.release()

    def mjpeg_stream() -> Generator[bytes, None, None]:
        def open_camera():
            if os.name == "nt":
                return cv2.VideoCapture(0, cv2.CAP_DSHOW)
            return cv2.VideoCapture(0)

        cap = open_camera()

        # Try to minimize latency on webcam streams.
        live_width = int(os.environ.get("LIVE_WIDTH", "640"))
        live_height = int(os.environ.get("LIVE_HEIGHT", "480"))
        jpeg_quality = int(os.environ.get("LIVE_JPEG_QUALITY", "70"))
        infer_every = max(1, int(os.environ.get("LIVE_INFER_EVERY", "2")))
        drop_grabs = max(0, int(os.environ.get("LIVE_DROP_GRABS", "2")))
        imgsz = int(os.environ.get("LIVE_IMGSZ", "640"))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, live_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, live_height)
        # Some backends honor this and reduce buffering.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        conf_thres = float(os.environ.get("CONF_THRES", "0.25"))

        last_jpeg = None
        last_defect = False
        last_score = 0.0
        frame_idx = 0
        fail_count = 0
        while True:
            # Drop a few buffered frames so we stay close to "real time".
            for _ in range(drop_grabs):
                cap.grab()

            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                # Temporary hiccups happen; try to recover a bit.
                if fail_count < 30:
                    time.sleep(0.05)
                    continue
                # Hard reset the capture.
                cap.release()
                time.sleep(0.2)
                cap = open_camera()
                fail_count = 0
                continue
            fail_count = 0

            # Throttle inference to reduce lag (reuse last annotated frame in between).
            if frame_idx % infer_every == 0 or last_jpeg is None:
                try:
                    boxed, last_defect, last_score = predict_and_annotate(frame, conf_thres)

                    _set_live_state(last_defect, last_score)
                except Exception:
                    boxed = frame
                    _set_live_state(False, 0.0)

                ok, enc = cv2.imencode(
                    ".jpg",
                    boxed,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )
                if ok:
                    last_jpeg = enc.tobytes()

            frame_idx += 1

            if last_jpeg is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + last_jpeg
                + b"\r\n"
            )

            # Tiny sleep to avoid pegging a CPU core when camera FPS is low.
            time.sleep(0.001)
        cap.release()

    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/stop-live")
def stop_live_legacy():
    # The MJPEG stream stops client-side when the <img> src is cleared.
    return "stopped"


# Preload models at startup and run a warmup inference so the first real
# request doesn't incur extra latency (or trigger Render's 30-s proxy timeout).
def _preload_models():
    t0 = time.time()
    try:
        models = get_models()
        app.logger.info(
            "Models preloaded in %.1fs: %s",
            time.time() - t0,
            ", ".join(m["name"] for m in models),
        )
    except Exception as e:
        app.logger.exception("Failed to preload models: %s", e)
        return

    # Warmup: run a tiny dummy image through each model so ONNX Runtime
    # allocates its internal buffers *before* the first real request.
    try:
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        imgsz = int(os.environ.get("MODEL_IMGSZ", "640"))
        for entry in models:
            m, mtype = entry["model"], entry["type"]
            if mtype == "onnxrt":
                _predict_boxes_onnxrt(m, dummy, 0.9, imgsz)
            elif mtype == "yolov5":
                _predict_boxes_yolov5(m, dummy, 0.9, imgsz, None)
            else:
                _predict_boxes_yolov8(m, dummy, 0.9, imgsz, None)
        app.logger.info("Warmup inference complete (%.1fs total)", time.time() - t0)
    except Exception as e:
        app.logger.warning("Warmup inference failed (non-fatal): %s", e)


# Run preload when not in debug/reload mode
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not os.environ.get("FLASK_DEBUG"):
    _preload_models()


if __name__ == "__main__":
    # Render (and many PaaS) provide the HTTP port via the PORT env var.
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
