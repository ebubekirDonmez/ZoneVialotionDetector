"""
ROI Checker - Polygon İçinde Mi Kontrolü
=========================================

Kişilerin ROI (Region of Interest) içinde olup olmadığını kontrol eder.

Kullanım:
    from modules.roi_checker import ROIChecker
    
    # JSON'dan yükle
    checker = ROIChecker.load_from_json('data/roi_zones.json')
    
    # Nokta kontrol et
    is_inside, zone_name = checker.check_point(x=150, y=200)
    
    # Bbox kontrol et
    bbox = (100, 150, 200, 300)  # (x1, y1, x2, y2)
    is_inside, zone_name = checker.check_bbox(bbox, method='bottom_center')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


# Tip tanımlamaları
Point = Tuple[int, int]              # (x, y)
Polygon = List[Point]                # [(x1,y1), (x2,y2), ...]
BBox = Tuple[int, int, int, int]     # (x1, y1, x2, y2)


class ROIChecker:
    """
    ROI kontrol sınıfı - polygon içinde mi kontrolü.
    
    Bu sınıf şunları yapar:
    1. Bir noktanın polygon içinde olup olmadığını kontrol eder
    2. Bir bbox'ın (kişinin) polygon içinde olup olmadığını kontrol eder
    3. ROI'leri görselleştirir (frame üzerine çizer)
    """
    
    def __init__(self, roi_zones: dict[str, Polygon]):
        """
        Args:
            roi_zones: {zone_name: [(x1,y1), (x2,y2), ...]} formatında ROI'ler
            
        Örnek:
            roi_zones = {
                "zone_1": [(100, 100), (300, 100), (300, 300), (100, 300)],
                "zone_2": [(400, 200), (600, 200), (600, 400), (400, 400)]
            }
        """
        self.roi_zones = roi_zones
        
        # Her ROI için numpy array'e çevir (performans için)
        # OpenCV'nin pointPolygonTest fonksiyonu numpy array bekler
        self.roi_arrays = {
            zone_name: np.array(points, dtype=np.int32)
            for zone_name, points in roi_zones.items()
        }
    
    def check_point(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """
        Bir noktanın herhangi bir ROI içinde olup olmadığını kontrol et.
        
        Mantık:
        - Tüm ROI'leri teker teker kontrol eder
        - İlk eşleşen ROI'yi döndürür
        - Hiçbir ROI'de değilse (False, None) döndürür
        
        Args:
            x, y: Kontrol edilecek nokta koordinatları
            
        Returns:
            (is_inside, zone_name): 
                - ROI içindeyse → (True, "zone_1")
                - Hiçbir ROI'de değilse → (False, None)
                
        Örnek:
            is_inside, zone_name = checker.check_point(150, 200)
            if is_inside:
                print(f"Nokta {zone_name} içinde!")
        """
        point = (int(x), int(y))
        
        # Tüm ROI'leri kontrol et
        for zone_name, polygon_array in self.roi_arrays.items():
            # OpenCV pointPolygonTest kullan
            # Pozitif = içerde, negatif = dışarıda, 0 = sınırda
            result = cv2.pointPolygonTest(polygon_array, point, measureDist=False)
            
            if result >= 0:  # İçerde veya sınırda
                return True, zone_name
        
        # Hiçbir ROI'de değil
        return False, None
    
    def check_bbox(self, bbox: BBox, method: str = 'bottom_center') -> Tuple[bool, Optional[str]]:
        """
        Bir bounding box'ın ROI içinde olup olmadığını kontrol et.
        
        BBox (Bounding Box) = Kişinin etrafındaki dikdörtgen
        
        Args:
            bbox: (x1, y1, x2, y2) formatında bbox
                x1, y1 = Sol üst köşe
                x2, y2 = Sağ alt köşe
                
            method: Kontrol yöntemi (hangisine bakacağız?)
                - 'bottom_center': Bbox alt ortası (ayak noktası) ← ÖNERİLEN
                - 'center': Bbox merkezi
                - 'any': Bbox'ın herhangi bir köşesi içerdeyse
                - 'all': Bbox'ın tüm köşeleri içerdeyse
                
        Returns:
            (is_inside, zone_name): ROI içindeyse (True, zone_adı), değilse (False, None)
            
        Örnek:
            bbox = (100, 150, 200, 300)
            is_inside, zone = checker.check_bbox(bbox, method='bottom_center')
            if is_inside:
                print(f"Kişi {zone} içinde!")
        """
        x1, y1, x2, y2 = bbox
        
        if method == 'bottom_center':
            # Ayak noktası (bbox alt ortası) - en güvenilir
            # Kişinin nerede durduğu önemli!
            point = (int((x1 + x2) / 2), int(y2))
            return self.check_point(*point)
        
        elif method == 'center':
            # Bbox merkezi
            point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            return self.check_point(*point)
        
        elif method == 'any':
            # Herhangi bir köşe içerdeyse True
            corners = [
                (x1, y1),  # Sol üst
                (x2, y1),  # Sağ üst
                (x1, y2),  # Sol alt
                (x2, y2),  # Sağ alt
            ]
            for corner in corners:
                is_inside, zone_name = self.check_point(*corner)
                if is_inside:
                    return True, zone_name
            return False, None
        
        elif method == 'all':
            # Tüm köşeler içerdeyse True
            corners = [
                (x1, y1),
                (x2, y1),
                (x1, y2),
                (x2, y2),
            ]
            zone_name = None
            for corner in corners:
                is_inside, zone_name = self.check_point(*corner)
                if not is_inside:
                    return False, None  # Bir köşe dışarıdaysa False
            # Hepsi içerdeyse True (son zone_name'i dön)
            return True, zone_name
        
        else:
            raise ValueError(f"Geçersiz method: {method}. Seçenekler: bottom_center, center, any, all")
    
    def get_zones_for_bbox(self, bbox: BBox, method: str = 'bottom_center') -> List[str]:
        """
        Bir bbox'ın içinde bulunduğu tüm ROI'leri döndür.
        (Çakışan ROI'ler için kullanışlı)
        
        Not: check_bbox() sadece ilk eşleşen zone'u döndürür.
             Bu fonksiyon ise TÜM eşleşen zone'ları döndürür.
        
        Args:
            bbox: (x1, y1, x2, y2)
            method: 'bottom_center' veya 'center'
            
        Returns:
            [zone_name1, zone_name2, ...]: İçinde bulunduğu ROI isimleri
            
        Örnek:
            zones = checker.get_zones_for_bbox(bbox)
            # → ["zone_1", "zone_2"]  (iki ROI'ye de giriyor)
        """
        x1, y1, x2, y2 = bbox
        
        if method == 'bottom_center':
            point = (int((x1 + x2) / 2), int(y2))
        elif method == 'center':
            point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        else:
            raise ValueError(f"get_zones_for_bbox sadece 'bottom_center' veya 'center' destekler")
        
        zones = []
        for zone_name, polygon_array in self.roi_arrays.items():
            result = cv2.pointPolygonTest(polygon_array, point, measureDist=False)
            if result >= 0:  # İçerde
                zones.append(zone_name)
        
        return zones
    
    def draw_roi_zones(
        self,
        frame: np.ndarray,
        fill_alpha: float = 0.2,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        ROI bölgelerini frame üzerine çiz (görselleştirme).
        
        Şunları çizer:
        1. Şeffaf dolgu (fill_alpha ile)
        2. Kenar çizgileri
        3. Zone isimleri (show_labels=True ise)
        
        Args:
            frame: Üzerine çizilecek frame
            fill_alpha: Dolgu şeffaflığı (0-1)
                0 = tamamen şeffaf
                1 = tamamen opak
            show_labels: ROI isimlerini göster
            
        Returns:
            Çizilmiş frame
            
        Örnek:
            frame = checker.draw_roi_zones(frame, fill_alpha=0.3)
            cv2.imshow('Frame', frame)
        """
        output = frame.copy()
        
        for idx, (zone_name, points) in enumerate(self.roi_zones.items()):
            color = self._get_zone_color(idx)
            
            # 1) Polygon'u doldur (şeffaf overlay)
            overlay = output.copy()
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color, cv2.LINE_AA)
            
            # Şeffaflık uygula (alpha blending)
            cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)
            
            # 2) Kenarları çiz (smooth çizgiler)
            cv2.polylines(output, [pts], True, color, 2, cv2.LINE_AA)
            
            # 3) İsim etiketi — küçük köşe tag'i (görüş alanını engellemiyor)
            if show_labels:
                pts_np = np.array(points)
                
                # Poligonun sol-üst köşesine yakın pozisyon
                min_x = int(np.min(pts_np[:, 0]))
                min_y = int(np.min(pts_np[:, 1]))
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.38
                font_thickness = 1
                
                (tw, th), baseline = cv2.getTextSize(
                    zone_name, font, font_scale, font_thickness
                )
                
                pad = 4
                tag_x = min_x + 6
                tag_y = min_y + th + pad + 6
                
                # Yarı saydam renkli arka plan
                tag_overlay = output.copy()
                cv2.rectangle(
                    tag_overlay,
                    (tag_x - pad, tag_y - th - pad),
                    (tag_x + tw + pad, tag_y + pad + baseline),
                    color,
                    -1
                )
                cv2.addWeighted(tag_overlay, 0.65, output, 0.35, 0, output)
                
                # Beyaz metin
                cv2.putText(
                    output,
                    zone_name,
                    (tag_x, tag_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )
        
        return output
    
    def _get_zone_color(self, index: int) -> Tuple[int, int, int]:
        """Zone index'ine göre renk döndür."""
        colors = [
            (0, 255, 0),    # Yeşil
            (255, 100, 0),  # Mavi
            (0, 100, 255),  # Turuncu
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Sarı
        ]
        return colors[index % len(colors)]
    
    @staticmethod
    def load_from_json(json_path: str) -> ROIChecker:
        """
        JSON dosyasından ROI'leri yükle ve ROIChecker oluştur.
        
        JSON formatı:
        {
            "zone_1": [[100, 100], [300, 100], [300, 300], [100, 300]],
            "zone_2": [[400, 200], [600, 200], [600, 400], [400, 400]]
        }
        
        Args:
            json_path: roi_zones.json dosyasının yolu
            
        Returns:
            ROIChecker instance
            
        Örnek:
            checker = ROIChecker.load_from_json('data/roi_zones.json')
        """
        if not Path(json_path).exists():
            raise FileNotFoundError(f"ROI dosyası bulunamadı: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            roi_zones = json.load(f)
        
        # JSON'dan tuple'a çevir
        # JSON: [[100, 100], [300, 100]] → Python: [(100, 100), (300, 100)]
        roi_zones_tuples = {
            zone_name: [tuple(p) for p in points]
            for zone_name, points in roi_zones.items()
        }
        
        return ROIChecker(roi_zones_tuples)


# ═══════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR (Dışarıdan da kullanılabilir)
# ═══════════════════════════════════════════════════════════════

def get_bbox_bottom_center(bbox: BBox) -> Point:
    """
    Bbox'ın alt orta noktasını (ayak noktası) döndür.
    
    Args:
        bbox: (x1, y1, x2, y2)
        
    Returns:
        (x, y): Alt orta nokta
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def get_bbox_center(bbox: BBox) -> Point:
    """
    Bbox'ın merkez noktasını döndür.
    
    Args:
        bbox: (x1, y1, x2, y2)
        
    Returns:
        (x, y): Merkez nokta
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def is_point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Basit polygon içinde mi kontrolü.
    
    Args:
        point: (x, y) kontrol edilecek nokta
        polygon: [(x1,y1), (x2,y2), ...] polygon noktaları
        
    Returns:
        True: İçerde, False: Dışarıda
        
    Örnek:
        polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        is_inside = is_point_in_polygon((50, 50), polygon)  # → True
    """
    polygon_array = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon_array, point, measureDist=False)
    return result >= 0
