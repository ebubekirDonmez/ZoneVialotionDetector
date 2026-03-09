"""
Zone Violation Detection - Ana Pipeline
========================================

Tüm modülleri birleştirerek alan ihlali tespiti yapar.

Kullanım:
    python main.py --video sample.mp4
    python main.py --video 0  # Kamera için
    python main.py --config custom_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# Kendi modüllerimiz
from modules.detector import select_device, load_yolo_model, predict_people
from modules.tracker import create_deepsort_tracker, yolo_results_to_deepsort_detections
from modules.roi_checker import ROIChecker
from modules.violation_engine import ViolationEngine
from modules.notifier import TelegramNotifier


class ZoneViolationDetector:
    """
    Ana alan ihlali tespit sistemi.
    
    Bu sınıf tüm modülleri birleştirir:
    1. YOLO Detector (kişi tespiti)
    2. DeepSORT Tracker (ID atama)
    3. ROI Checker (polygon kontrolü)
    4. Violation Engine (ihlal tespiti)
    """
    
    def __init__(self, config_path: str = 'config.yaml', roi_file: Optional[str] = None):
        """
        Args:
            config_path: Konfigürasyon dosyası yolu
            roi_file: ROI dosya yolu (belirtilmezse config'den alınır)
        """
        # Config yükle
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ROI dosya yolunu ayarla (parametre varsa override et)
        if roi_file:
            self.config['roi']['zones_file'] = roi_file
        
        print("=" * 60)
        print("🚀 ZONE VIOLATION DETECTION SYSTEM")
        print("=" * 60)
        
        # Modülleri başlat
        self._init_modules()
        
        # Çıktı klasörleri
        self._setup_output_dirs()
        
        # Performans metrikleri
        self.frame_count = 0
        self.total_detections = 0
        self.total_violations = 0
        
        # Uyarı gösterimi için
        self.last_violation_frame = -999
        self.violation_alert_duration = 90  # frames (3 saniye @ 30fps)
        
        # Resize ayarları
        self.resize_scale = self.config.get('video', {}).get('resize_scale', 1.0)
        if self.resize_scale < 1.0:
            print(f"⚡ Frame resize aktif: %{int(self.resize_scale * 100)} (~{1/(self.resize_scale**2):.1f}x hızlanma)")
        
        print("=" * 60)
        print("✅ Sistem hazır!")
        print("=" * 60 + "\n")
    
    def _init_modules(self):
        """Tüm modülleri başlat."""
        print("\n📦 Modüller yükleniyor...")
        
        # 1) YOLO Detector
        print("\n1️⃣ YOLO Detector başlatılıyor...")
        yolo_config = self.config['yolo']

        prefer_cuda = yolo_config.get('device', 'cuda') == 'cuda'
        self.device = select_device(prefer_cuda=prefer_cuda)

        model_path = yolo_config.get('model', 'yolov8n.pt')
        self.yolo_model = load_yolo_model(model_path, device=self.device, fuse=True)

        self.yolo_conf  = yolo_config.get('confidence', 0.5)
        self.yolo_iou   = yolo_config.get('iou_threshold', 0.45)
        self.yolo_imgsz = yolo_config.get('imgsz', 640)

        print(f"   ✅ YOLO hazır ({model_path} on {self.device.upper()})")

        # 2) DeepSORT Tracker
        print("\n2️⃣ DeepSORT Tracker başlatılıyor...")
        ds_config    = self.config['deepsort']
        embedder_gpu = ds_config.get('embedder_gpu', False)

        self.tracker = create_deepsort_tracker(
            max_age          = ds_config.get('max_age', 30),
            n_init           = ds_config.get('n_init', 3),
            max_iou_distance = ds_config.get('max_iou_distance', 0.7),
            embedder         = ds_config.get('embedder', 'mobilenet'),
            embedder_gpu     = embedder_gpu,
        )

        if not embedder_gpu:
            print(f"   ⚡ Embedder CPU modunda (GPU çakışması önleme)")

        print(f"   ✅ DeepSORT hazır")
        
        # 3) ROI Checker
        print("\n3️⃣ ROI Checker başlatılıyor...")
        roi_file = self.config['roi'].get('zones_file', 'data/roi_zones.json')
        
        if not Path(roi_file).exists():
            print(f"   ❌ ROI dosyası bulunamadı: {roi_file}")
            print(f"   💡 ROI tanımlamak için: python tools/roi_editor.py --video <video_yolu>")
            sys.exit(1)
        
        self.roi_checker = ROIChecker.load_from_json(roi_file)
        print(f"   ✅ {len(self.roi_checker.roi_zones)} ROI yüklendi")
        
        # 4) Violation Engine
        print("\n4️⃣ Violation Engine başlatılıyor...")
        self.violation_engine = ViolationEngine(self.config['violation_rules'])

        # 5) Telegram Notifier
        print("\n5️⃣ Telegram Notifier başlatılıyor...")
        self.notifier = TelegramNotifier(self.config)
    
    def _setup_output_dirs(self):
        """Çıktı klasörlerini oluştur."""
        Path("outputs/events").mkdir(parents=True, exist_ok=True)
        Path("outputs/videos").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
    
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Tek bir frame'i işle.
        
        İşlem akışı:
        1. Frame'i resize et (performans için)
        2. YOLO ile kişi tespit et
        3. DeepSORT ile ID ata
        4. ROI kontrolü yap
        5. Violation kontrolü yap
        6. Görselleştir
        
        Args:
            frame: İşlenecek frame (orijinal boyut)
            frame_idx: Frame numarası
            
        Returns:
            İşlenmiş (visualize edilmiş) frame (orijinal boyut)
        """
        self.frame_count += 1
        
        # ═══════════════════════════════════════
        # 0. Frame Resize (Performans Optimizasyonu)
        # ═══════════════════════════════════════
        original_frame = frame
        if self.resize_scale < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.resize_scale)
            new_h = int(h * self.resize_scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # ═══════════════════════════════════════
        # 1. YOLO Detection (Kişi Tespiti)
        # ═══════════════════════════════════════
        yolo_results = predict_people(
            self.yolo_model,
            frame,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            imgsz=self.yolo_imgsz,
            device=self.device,
            half=(self.device == 'cuda'),
        )

        self.total_detections += len(yolo_results.boxes)

        # ═══════════════════════════════════════
        # 2. DeepSORT Tracking (ID Atama)
        # ═══════════════════════════════════════
        detections = yolo_results_to_deepsort_detections(yolo_results)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # ═══════════════════════════════════════
        # 3. Görselleştirme Frame'i Hazırla
        # ═══════════════════════════════════════
        # Not: Görselleştirmeyi resize edilmiş frame üzerinde yapıyoruz
        output_frame = frame.copy()
        
        # ROI'leri çiz (scaled frame üzerinde, ROI koordinatlarını da scale et)
        if self.config['roi'].get('display_zones', True):
            # Eğer resize yaptıysak, ROI'leri de scale etmeliyiz
            if self.resize_scale < 1.0:
                # Geçici olarak scaled ROI oluştur
                scaled_roi_zones = {}
                for zone_name, points in self.roi_checker.roi_zones.items():
                    scaled_points = [(int(x * self.resize_scale), int(y * self.resize_scale)) 
                                   for x, y in points]
                    scaled_roi_zones[zone_name] = scaled_points
                
                # Geçici ROIChecker oluştur
                from modules.roi_checker import ROIChecker
                temp_checker = ROIChecker(scaled_roi_zones)
                output_frame = temp_checker.draw_roi_zones(
                    output_frame,
                    fill_alpha=self.config['roi'].get('fill_alpha', 0.2),
                    show_labels=self.config['roi'].get('show_labels', True)
                )
            else:
                output_frame = self.roi_checker.draw_roi_zones(
                    output_frame,
                    fill_alpha=self.config['roi'].get('fill_alpha', 0.2),
                    show_labels=self.config['roi'].get('show_labels', True)
                )
        
        # ═══════════════════════════════════════
        # 4. Her Track için ROI + Violation Kontrolü
        # ═══════════════════════════════════════
        for track in tracks:
            if not track.is_confirmed():
                continue  # Confirmed olmayan track'leri atla (n_init kontrolü)
            
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            bbox = (x1, y1, x2, y2)
            
            # Not: Bbox koordinatları resize edilmiş frame'de, ROI ise orijinal boyutta
            # ROI koordinatlarını scale etmeliyiz (eğer resize yaptıysak)
            if self.resize_scale < 1.0:
                # Bbox'ı orijinal boyuta scale et
                scale_factor = 1.0 / self.resize_scale
                x1_scaled = int(x1 * scale_factor)
                y1_scaled = int(y1 * scale_factor)
                x2_scaled = int(x2 * scale_factor)
                y2_scaled = int(y2 * scale_factor)
                bbox_for_roi = (x1_scaled, y1_scaled, x2_scaled, y2_scaled)
            else:
                bbox_for_roi = bbox
            
            # ROI kontrolü (scaled bbox ile)
            check_method = self.config['roi'].get('check_method', 'bottom_center')
            is_in_roi, zone_name = self.roi_checker.check_bbox(bbox_for_roi, method=check_method)
            
            # Violation Engine'e güncelle
            violation_data = self.violation_engine.update_track(
                track_id=track_id,
                in_roi=is_in_roi,
                zone_name=zone_name
            )
            
            # Yeni ihlal tespit edildiyse
            if violation_data is not None:
                self.total_violations += 1
                self.last_violation_frame = self.frame_count  # Uyarı gösterimi için

                # Ekran görüntüsü kaydet
                if self.config['violation_logging'].get('save_snapshots', True):
                    self._save_snapshot(output_frame, violation_data)

                # Telegram bildirimi — orijinal çözünürlük + orijinal bbox koordinatları
                self.notifier.notify(original_frame, bbox_for_roi, violation_data)
            
            # Track'i çiz
            self._draw_track(output_frame, track, is_in_roi, zone_name)
        
        # ═══════════════════════════════════════
        # 5. İstatistikleri Çiz
        # ═══════════════════════════════════════
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]
        self._draw_stats(output_frame, len(confirmed_tracks))
        
        # ═══════════════════════════════════════
        # 6. Periyodik Temizlik
        # ═══════════════════════════════════════
        if self.frame_count % 100 == 0:
            self.violation_engine.cleanup_old_tracks()
        
        # Not: Resize edilmiş frame'i döndürüyoruz (hem hız hem pencere boyutu için)
        return output_frame
    
    def _draw_track(
        self,
        frame: np.ndarray,
        track,
        is_in_roi: bool,
        zone_name: Optional[str]
    ):
        """Track'i frame üzerine çiz."""
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id
        
        # Modern renkler (ROI içindeyse parlak kırmızı, değilse cyan)
        color = (60, 60, 255) if is_in_roi else (255, 200, 80)  # Parlak kırmızı / Cyan
        thickness = 2 if is_in_roi else 1
        
        # Bbox çiz (smooth çizgiler için LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        
        # Track bilgisi al
        track_info = self.violation_engine.get_track_info(track_id)
        
        # Label oluştur
        label_lines = [f"ID: {track_id}"]
        
        if is_in_roi:
            label_lines.append(f"Zone: {zone_name}")
            if track_info:
                time_in_roi = track_info['time_in_roi']
                label_lines.append(f"Time: {time_in_roi:.1f}s")
                
                if track_info['violation_detected']:
                    label_lines.append("⚠️ VIOLATION!")
        
        # Label çiz (multi-line, bbox üstünde) - Modern tasarım
        y_offset = y1 - 12
        for line in reversed(label_lines):
            # Arka plan boyutu
            (label_w, label_h), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1
            )
            
            padding_x = 10
            padding_y = 6
            
            # Gölge efekti (koyu arka plan)
            cv2.rectangle(
                frame,
                (x1 + 2, y_offset - label_h - padding_y + 2),
                (x1 + label_w + padding_x + 2, y_offset + 2),
                (0, 0, 0),
                -1,
                cv2.LINE_AA
            )
            
            # Ana arka plan (renkli)
            cv2.rectangle(
                frame,
                (x1, y_offset - label_h - padding_y),
                (x1 + label_w + padding_x, y_offset),
                color,
                -1,
                cv2.LINE_AA
            )
            
            # Metin (anti-aliasing ile)
            cv2.putText(
                frame,
                line,
                (x1 + padding_x // 2, y_offset - padding_y // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_offset -= (label_h + padding_y + 4)
    
    def _draw_stats(self, frame: np.ndarray, active_tracks: int):
        """İstatistikleri frame üzerine çiz — kompakt üst şerit."""
        h, w = frame.shape[:2]
        
        # Kompakt panel yüksekliği
        panel_h = 28
        
        # Yarı saydam koyu şerit (tek renk, döngüsüz → hızlı)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        
        # Alt ince accent çizgisi
        cv2.line(frame, (0, panel_h), (w, panel_h), (0, 180, 255), 1, cv2.LINE_AA)
        
        # ── Sol: Frame / Tracks / Violations tek satırda ──
        stats = [
            (f"Frame {self.frame_count}", (140, 255, 140)),
            (f"Tracks {active_tracks}", (255, 185, 80)),
            (f"Violations {self.total_violations}", (80, 100, 255)),
        ]
        x = 10
        for text, color in stats:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, text, (x, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            x += tw + 18
        
        # ── Sağ: Saat ──
        time_text = datetime.now().strftime('%H:%M:%S')
        (time_w, _), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, time_text, (w - time_w - 10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        
        # İhlal uyarı banner'ı (son ihlalden itibaren X frame göster)
        frames_since_violation = self.frame_count - self.last_violation_frame
        if frames_since_violation < self.violation_alert_duration:
            # Banner boyutları
            banner_h = 60
            banner_y = (h - banner_h) // 2
            
            # Yanıp sönen efekt (her 15 frame'de toggle)
            show_alert = (frames_since_violation // 15) % 2 == 0
            
            if show_alert:
                # Arka plan (kırmızı, yarı saydam)
                alert_overlay = frame.copy()
                cv2.rectangle(
                    alert_overlay,
                    (0, banner_y),
                    (w, banner_y + banner_h),
                    (0, 0, 180),
                    -1,
                    cv2.LINE_AA
                )
                cv2.addWeighted(alert_overlay, 0.7, frame, 0.3, 0, frame)
                
                # Üst ve alt border
                cv2.line(frame, (0, banner_y), (w, banner_y), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(frame, (0, banner_y + banner_h), (w, banner_y + banner_h), (0, 0, 255), 3, cv2.LINE_AA)
                
                # Uyarı metni
                alert_text = "VIOLATION DETECTED!"
                (alert_w, alert_h), _ = cv2.getTextSize(
                    alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2
                )
                alert_x = (w - alert_w) // 2
                
                # Metin gölgesi
                cv2.putText(
                    frame,
                    alert_text,
                    (alert_x + 2, banner_y + banner_h // 2 + alert_h // 2 + 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA
                )
                
                # Ana metin
                cv2.putText(
                    frame,
                    alert_text,
                    (alert_x, banner_y + banner_h // 2 + alert_h // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
    
    def _save_snapshot(self, frame: np.ndarray, violation_data: dict) -> str:
        """İhlal anını kaydet."""
        snapshot_dir = Path(self.config['violation_logging'].get('snapshot_dir', 'outputs/events'))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        track_id = violation_data['track_id']
        filename = f"violation_{timestamp}_track{track_id}.jpg"
        filepath = snapshot_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"   📸 Snapshot kaydedildi: {filepath}")
        
        return str(filepath)
    
    def run(self, video_source: str | int):
        """
        Ana işleme döngüsü.
        
        Args:
            video_source: Video dosyası yolu veya kamera ID
        """
        # Video aç
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Video açılamadı: {video_source}")
            return
        
        # Video bilgisi
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video bilgisi:")
        print(f"   Boyut: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Toplam frame: {total_frames}")
        print("\n🎬 İşleme başlıyor...")
        print("   [Q] Çıkış")
        print(f"   [T] Saat kuralı: {'AÇIK' if self.violation_engine.time_rule_enabled else 'KAPALI'}")
        print()
        
        # Resize için boyutları ayarla
        if self.resize_scale < 1.0:
            out_width = int(width * self.resize_scale)
            out_height = int(height * self.resize_scale)
        else:
            out_width = width
            out_height = height
        
        # Video writer (çıktı videosu için)
        output_path = Path("outputs/videos") / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
        
        # Görüntüleme çözünürlüğü (işleme çözünürlüğünden bağımsız)
        # process → out_width x out_height, display → display_w x display_h
        window_scale = self.config.get('video', {}).get('window_scale', 1.0)
        display_w = int(width * window_scale)
        display_h = int(height * window_scale)
        print(f"🖥️  Display: {display_w}x{display_h} (işleme: {out_width}x{out_height})")
        
        # Sabit pencere — AUTOSIZE ile yeniden boyutlandırma engellenir
        cv2.namedWindow('Zone Violation Detection', cv2.WINDOW_AUTOSIZE)
        
        # Frame skip ayarı
        process_every_n = self.config.get('video', {}).get('process_every_n_frames', 1)
        
        frame_idx = 0
        last_output_frame = None  # Frame skip için son işlenmiş frame'i sakla
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Frame skip (performans optimizasyonu)
                if frame_idx % process_every_n == 0:
                    # Bu frame'i işle
                    output_frame = self.process_frame(frame, frame_idx)
                    last_output_frame = output_frame
                else:
                    # Bu frame'i atla, önceki frame'i kullan
                    output_frame = last_output_frame if last_output_frame is not None else frame
                
                # Çıktı videosuna kaydet
                out.write(output_frame)
                
                # Göster
                if self.config.get('video', {}).get('display_window', True):
                    # İşleme boyutunu display boyutuna ölçekle (kaliteli görüntü)
                    if out_width != display_w or out_height != display_h:
                        display_frame = cv2.resize(
                            output_frame, (display_w, display_h),
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        display_frame = output_frame
                    cv2.imshow('Zone Violation Detection', display_frame)
                    
                    # Klavye kontrolleri
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        # Q: Çıkış
                        print("\n⏹️  Kullanıcı tarafından durduruldu")
                        break
                    elif key == ord('t'):
                        # T: Saat kuralını aç/kapat
                        self.violation_engine.toggle_time_rule()
                        status = "AÇIK" if self.violation_engine.time_rule_enabled else "KAPALI"
                        print(f"\n⏰ Saat kuralı: {status}")
                        if not self.violation_engine.time_rule_enabled:
                            print("   → Tüm ROI girişleri ihlal sayılacak")
                        else:
                            start = self.violation_engine.rule.authorized_start.strftime('%H:%M')
                            end = self.violation_engine.rule.authorized_end.strftime('%H:%M')
                            print(f"   → Sadece {start}-{end} dışı ihlal")
                
                frame_idx += 1
                
                # İlerleme göster (her 30 frame'de bir)
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                    print(f"⏳ İşleniyor: {frame_idx}/{total_frames} ({progress:.1f}%)")
        
        except KeyboardInterrupt:
            print("\n⚠️  İşlem kullanıcı tarafından iptal edildi")
        
        finally:
            # Temizlik
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Özet
            print("\n" + "=" * 60)
            print("✅ İŞLEM TAMAMLANDI")
            print("=" * 60)
            print(f"📊 Özet:")
            print(f"   Toplam frame: {self.frame_count}")
            print(f"   Toplam tespit: {self.total_detections}")
            print(f"   Toplam ihlal: {self.total_violations}")
            print(f"   Çıktı video: {output_path}")
            print("=" * 60 + "\n")


def get_roi_filepath(video_source) -> str:
    """
    Video yolundan ROI dosya yolu oluşturur.
    
    Args:
        video_source: Video dosya yolu veya kamera ID
        
    Returns:
        ROI JSON dosya yolu (örn: data/sample-1.mp4_roi.json)
    """
    if isinstance(video_source, int):
        # Kamera için
        return f"data/camera_{video_source}_roi.json"
    else:
        # Video dosyası için
        video_name = Path(video_source).name
        return f"data/{video_name}_roi.json"


def main():
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description='Zone Violation Detection - Alan ihlali tespit sistemi'
    )
    parser.add_argument(
        '--video',
        type=str,
        default='sample-1.mp4',
        help='Video dosyası yolu veya kamera ID (örn: 0)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Konfigürasyon dosyası yolu'
    )
    parser.add_argument(
        '--edit-roi',
        action='store_true',
        help='ROI editor\'u ac (video uzerinde bolge ciz)'
    )
    
    args = parser.parse_args()
    
    # Video source (sayı ise kamera, string ise dosya)
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    
    # Video için özel ROI dosya yolu
    roi_filepath = get_roi_filepath(video_source)
    
    # ROI düzenleme modu
    if args.edit_roi:
        print("=" * 60)
        print("ROI EDITOR MODU")
        print("=" * 60)
        print(f"Video: {video_source}")
        print(f"ROI dosyası: {roi_filepath}")
        print()
        
        # ROI editor'ı çalıştır
        try:
            # Tüm yolları absolute yap — subprocess farklı dizinde çalışabilir
            base_dir = Path(__file__).parent.resolve()
            editor_script = str(base_dir / 'tools' / 'roi_editor.py')
            video_abs = str((base_dir / str(video_source)).resolve()) \
                if not isinstance(video_source, int) else str(video_source)
            roi_abs = str((base_dir / roi_filepath).resolve())

            cmd = [
                sys.executable,
                editor_script,
                '--video', video_abs,
                '--output', roi_abs
            ]
            result = subprocess.run(cmd, check=True)
            
            if result.returncode != 0:
                print(f"❌ ROI editor hata verdi: {result.returncode}")
                return
                
            print("\n" + "=" * 60)
            print("ROI KAYDI TAMAMLANDI")
            print("=" * 60)
            
            # ROI dosyası kontrol
            if not os.path.exists(roi_filepath):
                print(f"⚠️ ROI dosyası oluşturulmadı: {roi_filepath}")
                print("ROI çizmeden çıkış yaptınız. Tekrar --edit-roi ile çalıştırın.")
                return
                
            # Kaydedilen ROI sayısı
            with open(roi_filepath, 'r', encoding='utf-8') as f:
                roi_data = json.load(f)
                print(f"✅ {len(roi_data)} ROI kaydedildi")
            print()
            
        except Exception as e:
            print(f"❌ ROI editor hatası: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # ROI dosyası kontrolü (edit modu değilse)
    if not args.edit_roi:
        if not os.path.exists(roi_filepath):
            print("=" * 60)
            print("⚠️ ROI DOSYASI BULUNAMADI")
            print("=" * 60)
            print(f"Video: {video_source}")
            print(f"Aranan ROI dosyası: {roi_filepath}")
            print()
            print("Lütfen önce ROI çizin:")
            print(f"  python main.py --video {video_source} --edit-roi")
            print("=" * 60)
            return
    
    # Sistemi başlat (ROI dosyası ile)
    try:
        print("=" * 60)
        print("İHLAL TESPİT SİSTEMİ BAŞLATILIYOR")
        print("=" * 60)
        print(f"Video: {video_source}")
        print(f"ROI dosyası: {roi_filepath}")
        print("=" * 60)
        print()
        
        system = ZoneViolationDetector(args.config, roi_file=roi_filepath)
        system.run(video_source)
    except FileNotFoundError as e:
        print(f"❌ Dosya bulunamadı: {e}")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
