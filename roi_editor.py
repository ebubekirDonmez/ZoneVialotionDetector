"""
ROI Editor - Polygon Çizim Aracı
=================================

Fare ile tıklayarak polygon ROI tanımlama aracı.

Kullanım:
    python tools/roi_editor.py --video sample.mp4
    python tools/roi_editor.py --video 0  # Kamera için

Kontroller:
    • Sol tık         : Nokta ekle
    • İlk noktaya yakın tık : Polygon'u kapat (otomatik)
    • ENTER/SPACE    : Mevcut polygon'u kaydet
    • U              : Son ROI'yi sil (Undo)
    • C              : Şu anki çizimi iptal et
    • S              : Kaydet ve çık
    • Q/ESC          : Kaydetmeden çık
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


# Tip tanımlamaları
Point = Tuple[int, int]
Polygon = List[Point]


class ROIEditor:
    """Polygon tabanlı ROI çizim aracı."""
    
    def __init__(self, video_source: str | int, output_path: str = 'data/roi_zones.json'):
        """
        Args:
            video_source: Video dosyası yolu veya kamera ID
            output_path: Çıktı JSON dosyası yolu
        """
        self.video_source = video_source
        self.output_path = output_path
        
        # Video'dan ilk frame'i al
        cap = cv2.VideoCapture(video_source)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Video okunamadı: {video_source}")
        
        self.frame_height, self.frame_width = self.frame.shape[:2]
        self.display_frame = self.frame.copy()
        
        # ROI durumu
        self.roi_zones: dict[str, Polygon] = {}  # Kaydedilmiş ROI'ler
        self.current_points: List[Point] = []    # Şu an çizilen polygon noktaları
        self.current_zone_name: Optional[str] = None
        self.is_drawing: bool = False             # Çizim yapılıyor mu?
        self.zone_counter: int = 1                # Zone sayacı
        
        # Çizim parametreleri
        self.point_radius = 6
        self.line_thickness = 2
        self.close_threshold = 20  # İlk noktaya bu kadar yakınsa otomatik kapat
        
        # Renkler (her ROI için farklı)
        self.colors = [
            (0, 255, 0),      # Yeşil
            (255, 100, 0),    # Mavi
            (0, 100, 255),    # Turuncu
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Sarı
        ]
        
        # Mouse hover için
        self.hover_point: Optional[Point] = None
    
    def get_color(self, index: int) -> Tuple[int, int, int]:
        """ROI index'ine göre renk döndür."""
        return self.colors[index % len(self.colors)]
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """
        OpenCV mouse event handler.
        
        Args:
            event: Mouse event tipi (MOUSEMOVE, LBUTTONDOWN, vs.)
            x, y: Mouse koordinatları
        """
        # Mouse hareket (hover efekti için)
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_point = (x, y)
            if self.is_drawing and len(self.current_points) > 0:
                self.update_display()
        
        # Sol tık
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not self.is_drawing:
                # Yeni ROI başlat
                self.start_new_roi(x, y)
            else:
                # Mevcut ROI'ye nokta ekle veya kapat
                self.add_point_or_close(x, y)
            
            self.update_display()
    
    def start_new_roi(self, x: int, y: int) -> None:
        """Yeni ROI çizmeye başla."""
        self.is_drawing = True
        self.current_zone_name = f"zone_{self.zone_counter}"
        self.current_points = [(x, y)]
        print(f"\n[NEW] Yeni ROI basladi: {self.current_zone_name}")
        print(f"   [POINT] Ilk nokta: ({x}, {y})")
        print(f"   [TIP] Ipucu: Ilk noktaya yakin tiklayarak kapatabilirsiniz")
    
    def add_point_or_close(self, x: int, y: int) -> None:
        """Nokta ekle veya polygon'u kapat."""
        first_point = self.current_points[0]
        distance = np.sqrt((x - first_point[0])**2 + (y - first_point[1])**2)
        
        # İlk noktaya yeterince yakın mı?
        if distance < self.close_threshold and len(self.current_points) >= 3:
            self.finish_roi()
        else:
            # Yeni nokta ekle
            self.current_points.append((x, y))
            print(f"   [POINT] Nokta {len(self.current_points)}: ({x}, {y})")
    
    def finish_roi(self) -> None:
        """Mevcut ROI'yi tamamla ve kaydet."""
        if len(self.current_points) >= 3:
            self.roi_zones[self.current_zone_name] = self.current_points.copy()
            print(f"[OK] ROI tamamlandi: {self.current_zone_name}")
            print(f"   [INFO] Toplam {len(self.current_points)} nokta")
            print(f"   [INFO] Toplam ROI sayisi: {len(self.roi_zones)}")
            
            self.zone_counter += 1
            self.is_drawing = False
            self.current_points = []
            self.current_zone_name = None
        else:
            print("[ERROR] En az 3 nokta gerekli!")
    
    def update_display(self) -> None:
        """Ekranı güncelle (tüm ROI'ler + mevcut çizim)."""
        self.display_frame = self.frame.copy()
        
        # 1) Kaydedilmiş ROI'leri çiz
        for idx, (zone_name, points) in enumerate(self.roi_zones.items()):
            self._draw_completed_roi(zone_name, points, idx)
        
        # 2) Şu an çizilen ROI'yi çiz
        if self.is_drawing and len(self.current_points) > 0:
            self._draw_current_roi()
        
        # 3) Bilgi paneli
        self._draw_info_panel()
    
    def _draw_completed_roi(self, zone_name: str, points: Polygon, idx: int) -> None:
        """Tamamlanmış bir ROI'yi çiz."""
        color = self.get_color(idx)
        
        # Polygon'u doldur (şeffaf overlay)
        overlay = self.display_frame.copy()
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, self.display_frame, 0.75, 0, self.display_frame)
        
        # Kenarları çiz (kalın)
        cv2.polylines(self.display_frame, [pts], True, color, self.line_thickness + 1)
        
        # Noktaları çiz
        for point in points:
            cv2.circle(self.display_frame, point, self.point_radius, color, -1)
            cv2.circle(self.display_frame, point, self.point_radius + 2, (255, 255, 255), 2)
        
        # İsim etiketi (ROI ortasına)
        centroid = np.mean(points, axis=0).astype(int)
        label_bg_size = cv2.getTextSize(zone_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            self.display_frame,
            (centroid[0] - label_bg_size[0]//2 - 5, centroid[1] - label_bg_size[1] - 5),
            (centroid[0] + label_bg_size[0]//2 + 5, centroid[1] + 5),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            self.display_frame,
            zone_name,
            (centroid[0] - label_bg_size[0]//2, centroid[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    
    def _draw_current_roi(self) -> None:
        """Şu an çizilmekte olan ROI'yi çiz."""
        color = self.get_color(self.zone_counter - 1)
        
        # Noktaları çiz
        for i, point in enumerate(self.current_points):
            # Nokta
            cv2.circle(self.display_frame, point, self.point_radius, color, -1)
            cv2.circle(self.display_frame, point, self.point_radius + 2, (255, 255, 255), 2)
            
            # Nokta numarası
            cv2.putText(
                self.display_frame,
                str(i + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # Çizgileri çiz (nokta arası)
        for i in range(len(self.current_points) - 1):
            cv2.line(
                self.display_frame,
                self.current_points[i],
                self.current_points[i + 1],
                color,
                self.line_thickness
            )
        
        # Hover preview çizgisi (son nokta → mouse → ilk nokta)
        if len(self.current_points) >= 2 and self.hover_point:
            cv2.line(
                self.display_frame,
                self.current_points[-1],
                self.hover_point,
                color,
                1,
                cv2.LINE_AA
            )
            cv2.line(
                self.display_frame,
                self.hover_point,
                self.current_points[0],
                color,
                1,
                cv2.LINE_AA
            )
        
        # İlk noktanın yakınlık alanı (kapatma için)
        if len(self.current_points) >= 3:
            first_point = self.current_points[0]
            cv2.circle(
                self.display_frame,
                first_point,
                self.close_threshold,
                (0, 255, 255),
                1
            )
    
    def _draw_info_panel(self) -> None:
        """Bilgi paneli çiz (üst kısımda)."""
        panel_height = 120
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_frame, 0.3, 0, self.display_frame)
        
        # Başlık
        cv2.putText(
            self.display_frame,
            "ROI EDITOR",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        
        # Durum bilgisi
        status_lines = [
            f"ROI Sayisi: {len(self.roi_zones)}",
            f"Durum: {'CIZILIYOR...' if self.is_drawing else 'BEKLIYOR'}",
            f"Nokta: {len(self.current_points)}" if self.is_drawing else "",
        ]
        
        y = 50
        for line in status_lines:
            if line:
                cv2.putText(
                    self.display_frame,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y += 25
    
    def save_roi(self) -> None:
        """ROI'leri JSON'a kaydet."""
        # Output dizinini oluştur
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Kaydet
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.roi_zones, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"[SAVED] ROI'ler kaydedildi: {self.output_path}")
        print(f"   [INFO] Toplam {len(self.roi_zones)} bolge tanimlandi")
        
        # Her ROI'nin detaylarını yazdır
        for zone_name, points in self.roi_zones.items():
            print(f"\n   - {zone_name}:")
            print(f"      Nokta sayisi: {len(points)}")
            print(f"      Koordinatlar: {points[:3]}{'...' if len(points) > 3 else ''}")
        
        print(f"{'='*60}\n")
    
    def load_existing_roi(self) -> None:
        """Var olan ROI'leri yükle."""
        if Path(self.output_path).exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # JSON'dan tuple'a çevir
                self.roi_zones = {k: [tuple(p) for p in v] for k, v in loaded.items()}
            
            # Zone counter'ı güncelle
            if self.roi_zones:
                max_zone = max([
                    int(name.split('_')[1]) 
                    for name in self.roi_zones.keys() 
                    if name.startswith('zone_')
                ], default=0)
                self.zone_counter = max_zone + 1
            
            print(f"[INFO] Mevcut ROI'ler yuklendi: {len(self.roi_zones)} bolge")
        else:
            print("[INFO] Yeni ROI dosyasi olusturulacak")
    
    def delete_last_zone(self) -> None:
        """Son eklenen ROI'yi sil."""
        if self.roi_zones:
            last_zone = list(self.roi_zones.keys())[-1]
            del self.roi_zones[last_zone]
            self.zone_counter -= 1
            print(f"[UNDO] Silindi: {last_zone}")
            self.update_display()
        else:
            print("[WARN] Silinecek ROI yok")
    
    def cancel_current_drawing(self) -> None:
        """Mevcut çizimi iptal et."""
        if self.is_drawing:
            print(f"[CANCEL] Iptal edildi: {self.current_zone_name}")
            self.is_drawing = False
            self.current_points = []
            self.current_zone_name = None
            self.update_display()
    
    def print_instructions(self) -> None:
        """Kullanım talimatlarını yazdır."""
        print("\n" + "=" * 60)
        print("ROI EDITORU BASLATILDI")
        print("=" * 60)
        print("\nKULLANIM:")
        print("  - Sol tik           : Nokta ekle")
        print("  - Ilk noktaya yakin : Polygon'u otomatik kapat")
        print("  - ENTER/SPACE       : Mevcut polygon'u kaydet")
        print("  - U                 : Son ROI'yi sil (Undo)")
        print("  - C                 : Su anki cizimi iptal et")
        print("  - S                 : Kaydet ve cik")
        print("  - Q/ESC             : Kaydetmeden cik")
        print("\nIPUCU:")
        print("  - En az 3 nokta gerekli")
        print("  - Karmasik sekiller icin istediginiz kadar nokta ekleyin")
        print("  - Ilk noktanin etrafindaki sari cember = kapatma alani")
        print("=" * 60 + "\n")
    
    def run(self) -> None:
        """ROI editörünü başlat."""
        # Mevcut ROI'leri yükle
        self.load_existing_roi()
        self.update_display()
        
        # Mouse callback ayarla
        cv2.namedWindow('ROI Editor', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('ROI Editor', self.mouse_callback)
        
        # Talimatları yazdır
        self.print_instructions()
        
        # Ana döngü
        while True:
            cv2.imshow('ROI Editor', self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q veya ESC
                print("\n[CANCEL] Iptal edildi (kaydedilmedi)")
                break
            
            elif key == ord('s'):  # S - Save
                if self.is_drawing and len(self.current_points) >= 3:
                    self.finish_roi()
                
                if len(self.roi_zones) > 0:
                    self.save_roi()
                else:
                    print("[WARN] Hic ROI tanimlanmadi!")
                break
            
            elif key == 13 or key == 32:  # ENTER veya SPACE
                if self.is_drawing and len(self.current_points) >= 3:
                    self.finish_roi()
                    self.update_display()
                else:
                    print("[WARN] En az 3 nokta gerekli!")
            
            elif key == ord('u'):  # U - Undo
                self.delete_last_zone()
            
            elif key == ord('c'):  # C - Cancel
                self.cancel_current_drawing()
        
        cv2.destroyAllWindows()


def main() -> None:
    """Ana fonksiyon."""
    parser = argparse.ArgumentParser(
        description='ROI Tanımlama Aracı - Polygon çizim ile ROI oluşturun'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Video dosyası yolu veya kamera ID (örn: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/roi_zones.json',
        help='Çıktı JSON dosyası'
    )
    
    args = parser.parse_args()
    
    # Video source (sayı ise kamera, string ise dosya)
    try:
        video_source = int(args.video)
    except ValueError:
        # Relatif yolu absolute yap (çalışma dizininden bağımsız)
        video_source = str(Path(args.video).resolve())
    
    output_path = str(Path(args.output).resolve())

    # ROI Editor'ı başlat
    try:
        editor = ROIEditor(video_source, output_path)
        editor.run()
    except Exception as e:
        print(f"[ERROR] Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
