"""
detector.py
===========

Bu dosya "insan tespiti" (YOLO detection) modülüdür.

Amaç:
- `step1_yolo_detection.py` bir demo/script olarak kalırken,
  ana uygulama (main) tarafından import edilip çağrılabilecek temiz bir API sağlamak.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import torch
from ultralytics import YOLO


def select_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_yolo_model(model_path: str, device: str, fuse: bool = True) -> YOLO:
    """
    YOLO modelini yükle + cihaza taşı.
    Not: Bazı durumlarda fuse desteklenmeyebilir; korumalı çalışır.
    """
    model = YOLO(model_path)
    if fuse:
        try:
            model.fuse()
        except Exception:
            pass
    model.to(device)
    return model


def predict_people(
    model: YOLO,
    frame_bgr,
    *,
    conf: float,
    imgsz: int,
    max_det: int,
    device: str,
    half: bool,
):
    """Tek frame'de person tespiti yapar; ultralytics Results[0] döndürür."""
    results = model.predict(
        source=frame_bgr,
        conf=conf,
        classes=[0],  # person
        device=device,
        half=half,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )
    return results[0]


def run_on_video(
    video_path: str,
    *,
    model_path: str = "yolov8s.pt",
    prefer_cuda: bool = True,
    resize_scale: float = 1.0,
    confidence: float = 0.35,
    imgsz: int = 640,
    max_detections: int = 50,
    display_window: bool = True,
    window_name: str = "YOLO Detection - Step 1 (Resized)",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    İnsan tespiti demo akışı (video üstünde).
    Main kodunda sadece `video_path` vererek çağırabilirsin.
    """
    device = select_device(prefer_cuda=prefer_cuda)

    if verbose:
        print("\n" + "=" * 60)
        print("GPU KONTROL")
        print("=" * 60)

        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print("✅ GPU BULUNDU!")
            print(f"   Model: {gpu_name}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Device: {device}")
            print("   → GPU ile 5-10x hızlanma bekleniyor!")
        else:
            print("⚠️  GPU bulunamadı, CPU kullanılacak")
            print(f"   Device: {device}")

        print("=" * 60)

    if verbose:
        print(f"\n[1/4] YOLO modeli yükleniyor ({device.upper()})...")

    model = load_yolo_model(model_path, device=device, fuse=True)
    use_half = (device == "cuda")

    if verbose and use_half:
        print("   → Half Precision (FP16) aktif - 2x hızlanma bekleniyor")
    if verbose:
        print(f"✓ Model hazır ve {device.upper()}'da çalışıyor!")

    if verbose:
        print("\n[2/4] Video açılıyor...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print(f"❌ Hata: Video açılamadı - {video_path}")
        return {"ok": False, "error": "video_open_failed", "video_path": video_path}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print("✓ Video bilgileri:")
        print(f"  - Orijinal Çözünürlük: {width}x{height}")
        print(f"  - FPS: {fps}")
        print(f"  - Toplam frame: {total_frames}")

    new_width = int(width * resize_scale)
    new_height = int(height * resize_scale)
    if verbose:
        print("\n[3/4] Resize ayarı:")
        print(f"  - İşlenecek Çözünürlük: {new_width}x{new_height} ({int(resize_scale*100)}%)")
        print(
            f"  - Piksel azalması: {(1-resize_scale**2)*100:.0f}% → ~{1/(resize_scale**2):.1f}x hızlanma bekleniyor"
        )

    if verbose:
        print("\n[4/4] Tespit parametreleri:")
        print(f"  - Confidence: {confidence}")
        print("  - Sınıflar: [0] (person)")
        print(f"  - YOLO imgsz: {imgsz} (default: 640)")
        print(f"  - Max detections: {max_detections} (default: 300)")
        print(f"  → imgsz küçültme: ~{(640/imgsz)**2:.1f}x hızlanma bekleniyor!")

        print("\n[5/5] Video işleniyor... ('q' tuşu ile çık)")
        print("-" * 60)

    frame_count = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            if verbose:
                print("\n✓ Video bitti!")
            break

        frame_count += 1
        start_time = time.time()

        frame_resized = cv2.resize(frame, (new_width, new_height))

        detections = predict_people(
            model,
            frame_resized,
            conf=confidence,
            imgsz=imgsz,
            max_det=max_detections,
            device=device,
            half=use_half,
        )

        person_count = 0
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence_score = float(box.conf[0])
            person_count += 1

            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_resized,
                f"Person: {confidence_score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame_resized, (center_x, center_y), 4, (255, 0, 0), -1)

        process_time = time.time() - start_time
        total_time += process_time
        fps_inst = 1 / process_time if process_time > 0 else 0

        cv2.rectangle(frame_resized, (0, 0), (new_width, 115), (0, 0, 0), -1)
        info_lines = [
            f"Frame: {frame_count}/{total_frames}",
            f"Kisi sayisi: {person_count}",
            f"Device: {device.upper()}",
            f"FPS : {fps_inst:.1f}",
        ]
        y_offset = 25
        for line in info_lines:
            cv2.putText(
                frame_resized,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            y_offset += 22

        if display_window:
            cv2.imshow(window_name, frame_resized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                if verbose:
                    print("\n⚠️  Kullanıcı tarafından durduruldu")
                break

        if verbose and frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(
                f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                f"{person_count} kişi tespit edildi - {fps_inst:.1f} FPS"
            )

    cap.release()
    cv2.destroyAllWindows()

    final_fps = frame_count / total_time if total_time > 0 else 0
    if verbose:
        print("\n" + "=" * 60)
        print("ÖZET:")
        print(f"  - İşlenen frame: {frame_count}")
        print(f"  - Ortalama FPS: {final_fps:.1f}")
        print(f"  - Toplam süre: {total_time:.1f} saniye")
        print(f"  - Device: {device.upper()}")
        print("=" * 60)

    return {
        "ok": True,
        "video_path": video_path,
        "model_path": model_path,
        "device": device,
        "frames": frame_count,
        "avg_fps": final_fps,
    }


