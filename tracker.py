"""
tracker.py
==========

Bu dosya "tracking" (DeepSORT) modülüdür.

Amaç:
- YOLO ile tespit edilen kişilere ID atamak (DeepSORT).
- Ana uygulama tarafından import edilip çalıştırılabilecek bir API sunmak.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from detector import load_yolo_model, select_device


DeepsortDetection = Tuple[List[float], float, int]  # ([x, y, w, h], conf, cls)


def create_deepsort_tracker(
    *,
    max_age: int = 30,
    n_init: int = 3,
    max_iou_distance: float = 0.7,
    embedder: str = "mobilenet",
    embedder_gpu: bool = False,
) -> DeepSort:
    return DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_iou_distance=max_iou_distance,
        embedder=embedder,
        embedder_gpu=embedder_gpu,
    )


def yolo_results_to_deepsort_detections(results0) -> List[DeepsortDetection]:
    """
    Ultralytics Results[0] -> DeepSORT formatına çevir:
    DeepSORT formatı: ([x, y, w, h], confidence, class)
    """
    out: List[DeepsortDetection] = []
    for box in results0.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        out.append(([x, y, w, h], conf, cls))
    return out


def run_on_video(
    video_path: str,
    *,
    model_path: str = "yolov8n.pt",
    prefer_cuda: bool = True,
    confidence: float = 0.5,
    imgsz: int = 640,
    max_detections: int = 100,
    deepsort_max_age: int = 30,
    deepsort_n_init: int = 3,
    deepsort_max_iou_distance: float = 0.7,
    deepsort_embedder: str = "mobilenet",
    display_window: bool = True,
    window_name: str = "YOLO + DeepSORT Tracking",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    YOLO + DeepSORT tracking akışı.
    Dışarıdan main kodun çağırabilmesi için tek fonksiyonla paketlenmiştir.
    """
    device = select_device(prefer_cuda=prefer_cuda)
    use_half = (device == "cuda")

    if verbose:
        print("=" * 70)
        print("TRACKER MODÜL: YOLO + DeepSORT")
        print("=" * 70)
        print(f"Video: {video_path}")
        print(f"Device: {device.upper()}")

    # YOLO
    if verbose:
        print("\n[1/3] YOLO yükleniyor...")
    model: YOLO = load_yolo_model(model_path, device=device, fuse=True)
    if verbose and use_half:
        print("   → FP16 aktif")
    if verbose:
        print("✅ YOLO hazır!")

    # DeepSORT
    if verbose:
        print("\n[2/3] DeepSORT oluşturuluyor...")
    tracker = create_deepsort_tracker(
        max_age=deepsort_max_age,
        n_init=deepsort_n_init,
        max_iou_distance=deepsort_max_iou_distance,
        embedder=deepsort_embedder,
        embedder_gpu=(device == "cuda"),
    )
    if verbose:
        print("✅ DeepSORT hazır!")

    # Video
    if verbose:
        print("\n[3/3] Video açılıyor...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print("❌ Video açılamadı!")
        return {"ok": False, "error": "video_open_failed", "video_path": video_path}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"✅ Video: {width}x{height} @ {fps} FPS | total={total_frames}")
        print("\nVideo işleniyor... ('q' ile çık)")

    frame_count = 0
    start_time = time.time()
    max_track_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO detect
        results = model.predict(
            source=frame,
            conf=confidence,
            classes=[0],
            device=device,
            half=use_half,
            imgsz=imgsz,
            max_det=max_detections,
            verbose=False,
        )
        det0 = results[0]
        detections_deepsort = yolo_results_to_deepsort_detections(det0)

        # DeepSORT update
        tracks = tracker.update_tracks(detections_deepsort, frame=frame)

        confirmed_count = 0
        current_active_ids = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            confirmed_count += 1

            tid = t.track_id
            if tid > max_track_id:
                max_track_id = tid
            current_active_ids.append(tid)

            x1, y1, x2, y2 = map(int, t.to_ltrb())

            color = ((tid * 50) % 255, (tid * 100) % 255, (tid * 150) % 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID: {tid}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0

        # overlay
        cv2.rectangle(frame, (0, 0), (width, 110), (0, 0, 0), -1)
        info_lines = [
            f"Frame: {frame_count}/{total_frames}",
            f"FPS: {current_fps:.1f}",
            f"Confirmed Tracks: {confirmed_count}",
            f"Max ID: {max_track_id}",
        ]
        y = 25
        for line in info_lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25

        if display_window:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if verbose:
                    print("\n⚠️  Durduruldu")
                break

        if verbose and frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(
                f"Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                f"FPS: {current_fps:.1f} | Tracks: {confirmed_count}"
            )

    cap.release()
    cv2.destroyAllWindows()

    total_elapsed = time.time() - start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

    if verbose:
        print("\n" + "=" * 70)
        print("TRACKING ÖZETİ")
        print("=" * 70)
        print(f"Toplam Frame: {frame_count}")
        print(f"Ortalama FPS: {avg_fps:.1f}")
        print(f"Toplam Süre: {total_elapsed:.1f} saniye")
        print(f"Toplam Unique ID: {max_track_id}")
        print("=" * 70)

    return {
        "ok": True,
        "video_path": video_path,
        "model_path": model_path,
        "device": device,
        "frames": frame_count,
        "avg_fps": avg_fps,
        "max_track_id": max_track_id,
    }


