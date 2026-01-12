import io
import os
import shutil
import subprocess
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

from ultralytics import YOLO

app = Flask(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))

_model_lock = threading.Lock()
_predict_lock = threading.Lock()
_model = None


def get_model() -> YOLO:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            model_path = os.environ.get("MODEL_PATH") or os.path.join(ROOT, "model", "best.pt")
            _model = YOLO(model_path)
        return _model

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


def predict_and_annotate(frame: np.ndarray, conf_thres: float) -> tuple[np.ndarray, bool, float]:
    # Allow operators (e.g., Render) to reduce latency/memory via env vars.
    imgsz = int(os.environ.get("MODEL_IMGSZ", "416"))
    device = os.environ.get("MODEL_DEVICE")  # e.g. 'cpu' or '0'

    with _predict_lock:
        results = get_model().predict(
            source=frame,
            conf=conf_thres,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
    r0 = results[0]
    boxed = r0.plot()

    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return boxed, False, 0.0

    conf = boxes.conf
    max_conf = float(conf.max().item()) if conf is not None and len(conf) else 0.0
    return boxed, True, max_conf


@app.route("/")
def index():
    enable_live = os.environ.get("ENABLE_LIVE", "1").strip().lower() not in {"0", "false", "no"}
    return render_template("index.html", enable_live=enable_live)


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

    conf_thres = float(os.environ.get("CONF_THRES", "0.25"))
    try:
        boxed, defect, score = predict_and_annotate(img, conf_thres)
    except Exception as e:
        app.logger.exception("Inference failed for /upload-image")
        return {"error": f"Inference failed: {type(e).__name__}"}, 500

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
            return {"error": f"Inference failed: {type(e).__name__}"}, 500
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
                    with _predict_lock:
                        results = get_model().predict(
                            source=frame,
                            conf=conf_thres,
                            imgsz=imgsz,
                            device=os.environ.get("MODEL_DEVICE"),
                            verbose=False,
                        )
                    r0 = results[0]
                    boxed = r0.plot()

                    boxes = getattr(r0, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        conf = getattr(boxes, "conf", None)
                        last_score = float(conf.max().item()) if conf is not None and len(conf) else 0.0
                        last_defect = True
                    else:
                        last_score = 0.0
                        last_defect = False

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
