import shutil
import subprocess
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_android_onnx"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    (ROOT / "model" / "best.pt", "best.onnx"),
    (ROOT / "model" / "best(old).pt", "best_old.onnx"),
    (ROOT / "yolo11n.pt", "yolo11n.onnx"),
]


def _assert_single_file_onnx(onnx_path: Path) -> None:
    data_sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_sidecar.exists():
        raise RuntimeError(
            f"ONNX export created external data sidecar {data_sidecar}. "
            f"Android loads model bytes from assets, so we need a single-file ONNX." 
        )


def _export_with_yolov5_local(pt_path: Path, out_name: str) -> Path:
    # YOLOv5 export writes alongside the weights path, so export from a temp copy inside OUT_DIR.
    tmp_pt = OUT_DIR / (Path(out_name).stem + ".pt")
    if tmp_pt.exists():
        tmp_pt.unlink()
    shutil.copy2(pt_path, tmp_pt)

    export_py = ROOT / "yolov5_local" / "export.py"
    if not export_py.exists():
        raise FileNotFoundError(f"Missing YOLOv5 exporter: {export_py}")

    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(export_py),
        "--weights",
        str(tmp_pt),
        "--include",
        "onnx",
        "--img",
        "640",
        "--opset",
        "12",
        "--device",
        "cpu",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    tmp_onnx = tmp_pt.with_suffix(".onnx")
    if not tmp_onnx.exists():
        raise RuntimeError(f"YOLOv5 export did not produce: {tmp_onnx}")
    _assert_single_file_onnx(tmp_onnx)

    out_path = OUT_DIR / out_name
    shutil.move(str(tmp_onnx), str(out_path))
    tmp_pt.unlink(missing_ok=True)
    print(f"Saved: {out_path} ({out_path.stat().st_size/1024/1024:.2f} MB)")
    return out_path


def _export_with_ultralytics(pt_path: Path, out_name: str) -> Path:
    model = YOLO(str(pt_path))
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=False,
        dynamic=False,
    )
    exported_path = Path(exported_path)
    if not exported_path.exists():
        raise RuntimeError(f"Ultralytics export did not produce a file: {exported_path}")
    _assert_single_file_onnx(exported_path)

    out_path = OUT_DIR / out_name
    shutil.copy2(exported_path, out_path)
    print(f"Saved: {out_path} ({out_path.stat().st_size/1024/1024:.2f} MB)")
    return out_path


def export_one(pt_path: Path, out_name: str) -> Path:
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing: {pt_path}")

    print(f"\nExporting: {pt_path}")
    # best.pt and best(old).pt are YOLOv5-trained and not directly exportable with ultralytics YOLO.
    if pt_path.name.lower().startswith("best"):
        return _export_with_yolov5_local(pt_path, out_name)
    return _export_with_ultralytics(pt_path, out_name)


def main() -> None:
    produced = []
    for pt, out_name in MODELS:
        produced.append(export_one(pt, out_name))

    print("\nDone. Produced:")
    for p in produced:
        print("-", p)


if __name__ == "__main__":
    main()
