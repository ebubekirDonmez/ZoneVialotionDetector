"""
Violation Engine - İhlal Kuralları Motoru
==========================================

İhlal kurallarını uygular ve ihlal durumlarını yönetir.

Kurallar:
1. ROI kontrolü (kişi yasak bölgede mi?)
2. Zaman kontrolü (yetkisiz saatte mi?)

Kullanım:
    from modules.violation_engine import ViolationEngine
    
    engine = ViolationEngine(config)
    violation = engine.update_track(track_id, in_roi, zone_name)
"""

from __future__ import annotations

import time
from datetime import datetime, time as dt_time
from typing import Dict, Optional, Tuple


class ViolationRule:
    """
    İhlal kurallarını tanımlar ve kontrol eder.
    
    Bu sınıf şunları yapar:
    1. Config'den kuralları yükler (zaman aralıkları)
    2. "Şu an yetkisiz saat mi?" sorusunu cevaplar
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: violation_rules dict'i (config.yaml'dan gelir)
            
        Örnek config:
            {
                'enabled': True,
                'time_based': {
                    'enabled': True,
                    'start_time': '08:00',
                    'end_time': '18:00'
                }
            }
        """
        self.enabled = config.get('enabled', True)
        
        # Zaman kuralı
        self.time_based = config.get('time_based', {})
        self.time_based_enabled = self.time_based.get('enabled', False)
        
        if self.time_based_enabled:
            # Config'den saat aralığını al
            start_str = self.time_based.get('start_time', '08:00')  # "08:00"
            end_str = self.time_based.get('end_time', '18:00')      # "18:00"
            
            # String'i datetime.time objesine çevir
            start_h, start_m = map(int, start_str.split(':'))
            end_h, end_m = map(int, end_str.split(':'))
            
            self.authorized_start = dt_time(start_h, start_m)  # 08:00
            self.authorized_end = dt_time(end_h, end_m)        # 18:00
    
    def is_unauthorized_time(self, current_time: Optional[datetime] = None) -> bool:
        """
        Şu anki zaman yetkisiz mi?
        
        Args:
            current_time: Test için özel zaman (None ise şu anki sistem saati)
            
        Returns:
            True: Yetkisiz saat (ihlal), False: Yetkili saat (sorun yok)
            
        Örnek:
            # Şu an 22:00, yetkili saatler 08:00-18:00
            is_violation = rule.is_unauthorized_time()  # → True
        """
        if not self.time_based_enabled:
            return False  # Zaman kuralı kapalıysa hiçbir zaman ihlal yok
        
        if current_time is None:
            current_time = datetime.now()
        
        current = current_time.time()  # Sadece saat kısmını al (tarih değil)
        
        # Gece yarısından geçen saat aralığı kontrolü (örn: 22:00 - 06:00)
        if self.authorized_start > self.authorized_end:
            # Yetkili saat: 22:00'dan sonra VEYA 06:00'dan önce
            # Örnek: start=22:00, end=06:00
            # Yetkili: 22:00-23:59 ve 00:00-06:00
            authorized = current >= self.authorized_start or current <= self.authorized_end
        else:
            # Normal saat aralığı (örn: 08:00 - 18:00)
            # Yetkili: 08:00-18:00 arası
            authorized = self.authorized_start <= current <= self.authorized_end
        
        return not authorized  # Yetkili değilse True (ihlal var)


class TrackState:
    """
    Bir track'in (kişinin) durumunu takip eder.
    
    Bu sınıf her kişi için şunları saklar:
    - Ne zaman ilk görüldü?
    - ROI'de mi?
    - ROI'ye ne zaman girdi?
    - İhlal yaptı mı?
    """
    
    # Çıkış kararı vermeden önce beklenecek grace süresi (saniye).
    # Frame kırpılması veya tracking titremesinden kaynaklanan yanlış "çıkış"
    # sinyallerini bu süre boyunca yok sayarız.
    ROI_EXIT_GRACE = 1.5

    def __init__(self, track_id: int):
        """
        Args:
            track_id: Kişinin benzersiz ID'si (tracker'dan gelir)
        """
        self.track_id = track_id
        
        # Zaman bilgisi
        self.first_seen = time.time()
        self.last_seen = time.time()
        
        # ROI bilgisi
        self.in_roi = False
        self.roi_entry_time: Optional[float] = None
        self.current_zone: Optional[str] = None
        
        # Grace period: ROI'den çıktığı ilk an (None = içeride veya zaten dışarıda)
        self._roi_exit_start: Optional[float] = None
        
        # İhlal bilgisi
        self.violation_detected = False
        self.violation_reason = ""
    
    def update(self, in_roi: bool, zone_name: Optional[str] = None):
        """
        Track durumunu güncelle.

        Çıkış kararı hemen verilmez; ROI_EXIT_GRACE saniye boyunca kişi
        ROI dışında görünmeye devam ederse gerçek çıkış olarak kabul edilir.
        Bu sayede frame kırpılması veya kısa tracking gürültüsü nedeniyle
        yanlış çıkış tespiti ve zamanlayıcı sıfırlaması önlenir.

        Args:
            in_roi: ROI içinde mi?
            zone_name: Hangi zone (varsa)
        """
        now = time.time()
        self.last_seen = now

        if in_roi:
            # ── ROI içinde ──────────────────────────────────────────────────
            self._roi_exit_start = None  # Bekleyen çıkış sinyalini iptal et

            if not self.in_roi:
                # Yeni giriş (ya da grace süresi içinde geri döndü)
                if self.roi_entry_time is None:
                    self.roi_entry_time = now
                    print(f"   🔵 Track #{self.track_id} → {zone_name} içine girdi")
                self.in_roi = True

            self.current_zone = zone_name

        else:
            # ── ROI dışında ─────────────────────────────────────────────────
            if self.in_roi:
                # Grace period başlat (henüz çıkış kesinleşmedi)
                if self._roi_exit_start is None:
                    self._roi_exit_start = now

                if now - self._roi_exit_start >= self.ROI_EXIT_GRACE:
                    # Grace süresi doldu → gerçek çıkış
                    if self.roi_entry_time:
                        duration = now - self.roi_entry_time
                        print(f"   🔴 Track #{self.track_id} → "
                              f"{self.current_zone} dışına çıktı ({duration:.1f}s kaldı)")
                    self.in_roi = False
                    self.roi_entry_time = None
                    self.current_zone = None
                    self._roi_exit_start = None
                # else: grace süresindeyiz, in_roi=True olarak tut
    
    def get_time_in_roi(self) -> float:
        """
        ROI içinde geçirilen süre (saniye).
        
        Returns:
            ROI içinde kalma süresi (saniye)
            ROI'de değilse 0.0
        """
        if self.roi_entry_time is None:
            return 0.0
        return time.time() - self.roi_entry_time
    
    def get_lifetime(self) -> float:
        """
        Track'in toplam görünme süresi (saniye).
        
        Returns:
            İlk görülmeden şimdiye kadar geçen süre
        """
        return self.last_seen - self.first_seen


class ViolationEngine:
    """
    İhlal motoru - tüm kuralları uygular ve durumları yönetir.
    
    Bu sınıf şunları yapar:
    1. Tüm track'lerin durumlarını saklar
    2. Her frame'de track'leri günceller
    3. İhlal kurallarını kontrol eder
    4. Yeni ihlal tespit edilirse bildirim verir
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: config.yaml'dan gelen violation_rules dict'i
        """
        self.rule = ViolationRule(config)
        self.violation_rules = self.rule  # Erişim için alias
        
        # Track durumları (her track_id için bir TrackState)
        self.tracks: Dict[int, TrackState] = {}
        
        # İhlal ayarları
        self.min_time_in_roi = config.get('min_time_in_roi', 3.0)  # Minimum 3 saniye
        
        # Klavye ile toggle için
        self.time_rule_enabled = self.rule.time_based_enabled
        
        print(f"✅ Violation Engine başlatıldı")
        print(f"   ⏰ Zaman tabanlı: {'Aktif' if self.time_rule_enabled else 'Pasif'}")
        print(f"   ⏱️  Minimum ROI süresi: {self.min_time_in_roi}s (spam önleme)")
    
    def update_track(
        self,
        track_id: int,
        in_roi: bool,
        zone_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Track'i güncelle ve ihlal kontrolü yap.
        
        Bu fonksiyon her frame'de her track için çağrılır.
        
        Args:
            track_id: Track ID (tracker'dan gelir)
            in_roi: ROI içinde mi? (roi_checker'dan gelir)
            zone_name: Hangi ROI (varsa)
            
        Returns:
            Eğer YENİ bir ihlal tespit edildiyse violation_data dict'i
            Yoksa None
            
        Örnek:
            violation = engine.update_track(
                track_id=5,
                in_roi=True,
                zone_name="zone_1"
            )
            
            if violation:
                print(f"⚠️ İhlal: Track #{violation['track_id']}")
        """
        # Track yoksa oluştur
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id)
        
        track = self.tracks[track_id]
        track.update(in_roi, zone_name)
        
        # İhlal kontrolü yap
        if in_roi and track.get_time_in_roi() >= self.min_time_in_roi:
            # ROI'de yeterince uzun süre kaldıysa kontrol et
            
            # Yetkisiz saatte mi?
            # Eğer time_rule_enabled False ise → Her zaman unauthorized (tüm girişler ihlal)
            if self.time_rule_enabled:
                is_unauthorized = self.rule.is_unauthorized_time()
            else:
                is_unauthorized = True  # Saat kuralı kapalı → Tüm girişler ihlal
            
            if is_unauthorized and not track.violation_detected:
                # YENİ bir ihlal tespit edildi!
                track.violation_detected = True
                track.violation_reason = "Yetkisiz saatte ROI'de"
                
                # Violation data oluştur
                violation_data = {
                    'track_id': track_id,
                    'zone_name': zone_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'violation_type': 'Yetkisiz Saat',
                    'unauthorized_time': True,
                    'time_in_roi': track.get_time_in_roi(),
                }
                
                print(f"\n⚠️  ═══════════════════════════════════════")
                print(f"    İHLAL TESPİT EDİLDİ!")
                print(f"    ═══════════════════════════════════════")
                print(f"    🆔 Track ID: #{track_id}")
                print(f"    📍 Bölge: {zone_name}")
                print(f"    ⏰ Zaman: {violation_data['timestamp']}")
                print(f"    ⚖️  Sebep: Yetkisiz saatte ROI'de")
                print(f"    ⏱️  ROI süresi: {track.get_time_in_roi():.1f}s")
                print(f"    ═══════════════════════════════════════\n")
                
                return violation_data
        
        return None  # İhlal yok
    
    def cleanup_old_tracks(self, max_age: float = 30.0):
        """
        Eski track'leri temizle (bellek tasarrufu).
        
        Args:
            max_age: Track'in son görülme süresinden bu kadar saniye geçtiyse sil
            
        Not:
            Her 100 frame'de bir çağrılması önerilir.
        """
        current_time = time.time()
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        if to_remove:
            print(f"🧹 {len(to_remove)} eski track temizlendi")
    
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """
        Track bilgisini döndür (görselleştirme için).
        
        Args:
            track_id: Track ID
            
        Returns:
            Track bilgileri dict'i veya None
            
        Örnek:
            info = engine.get_track_info(5)
            if info:
                print(f"Track #{info['track_id']}: {info['time_in_roi']:.1f}s ROI'de")
        """
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        return {
            'track_id': track_id,
            'in_roi': track.in_roi,
            'current_zone': track.current_zone,
            'time_in_roi': track.get_time_in_roi(),
            'lifetime': track.get_lifetime(),
            'violation_detected': track.violation_detected,
            'violation_reason': track.violation_reason,
        }
    
    def get_stats(self) -> Dict:
        """
        Genel istatistikler döndür.
        
        Returns:
            {
                'total_tracks': int,
                'active_in_roi': int,
                'total_violations': int
            }
        """
        active_in_roi = sum(1 for t in self.tracks.values() if t.in_roi)
        total_violations = sum(1 for t in self.tracks.values() if t.violation_detected)
        
        return {
            'total_tracks': len(self.tracks),
            'active_in_roi': active_in_roi,
            'total_violations': total_violations,
        }
    
    def toggle_time_rule(self):
        """
        Saat kuralını aç/kapat (klavye kısayolu için).
        
        Çalışırken saat kuralını aktif/pasif yapabilirsin:
        - AÇIK: Sadece yetkisiz saatlerde ihlal sayılır
        - KAPALI: Tüm ROI girişleri ihlal sayılır
        """
        self.time_rule_enabled = not self.time_rule_enabled
    
    def reset(self):
        """Tüm track'leri ve durumları sıfırla."""
        self.tracks.clear()
        print("🔄 Violation Engine sıfırlandı")


# ═══════════════════════════════════════════════════════════════
# TEST FONKSİYONU (Geliştirme için)
# ═══════════════════════════════════════════════════════════════

def test_time_rules():
    """
    Zaman kurallarını test et.
    
    Kullanım:
        python -c "from modules.violation_engine import test_time_rules; test_time_rules()"
    """
    print("\n" + "="*60)
    print("ZAMAN KURALLARI TESTİ")
    print("="*60)
    
    # Test config
    config = {
        'enabled': True,
        'min_time_in_roi': 3.0,
        'time_based': {
            'enabled': True,
            'start_time': '08:00',
            'end_time': '18:00'
        }
    }
    
    rule = ViolationRule(config)
    
    # Farklı zamanları test et
    test_times = [
        datetime(2024, 1, 11, 7, 30),   # 07:30 - Yetkisiz
        datetime(2024, 1, 11, 8, 0),    # 08:00 - Yetkili (sınır)
        datetime(2024, 1, 11, 12, 0),   # 12:00 - Yetkili
        datetime(2024, 1, 11, 18, 0),   # 18:00 - Yetkili (sınır)
        datetime(2024, 1, 11, 18, 1),   # 18:01 - Yetkisiz
        datetime(2024, 1, 11, 22, 0),   # 22:00 - Yetkisiz
    ]
    
    print(f"\nYetkili saatler: {config['time_based']['start_time']} - {config['time_based']['end_time']}")
    print("\nTest sonuçları:")
    
    for test_time in test_times:
        is_violation = rule.is_unauthorized_time(test_time)
        status = "⚠️ YETKİSİZ" if is_violation else "✅ YETKİLİ"
        print(f"  {test_time.strftime('%H:%M')} → {status}")
    
    print("\nŞu anki sistem saati:")
    current = datetime.now()
    is_violation = rule.is_unauthorized_time()
    status = "⚠️ YETKİSİZ" if is_violation else "✅ YETKİLİ"
    print(f"  {current.strftime('%H:%M')} → {status}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    # Test için çalıştır
    test_time_rules()
