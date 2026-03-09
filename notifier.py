"""
Notifier - Telegram İhlal Bildirimi
=====================================

Alan ihlali tespit edildiğinde Telegram Bot API üzerinden
kişinin bbox kırpması + ihlal detaylarını gönderir.

Kullanım:
    config.yaml içinde:
        notifications:
          telegram:
            enabled: true
            bot_token: "123456:ABC..."
            chat_id: "123456789"
"""

from __future__ import annotations

import io
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests


class TelegramNotifier:
    """
    Telegram Bot API üzerinden ihlal bildirimi gönderir.

    Özellikler:
    - Non-blocking: Gönderim ayrı thread'de yapılır, ana döngü yavaşlamaz
    - Cooldown: Aynı kişi için tekrarlı bildirim engellenir
    - Kaliteli kırpma: Orijinal çözünürlük frame'den bbox kırpılır
    """

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, config: dict):
        """
        Args:
            config: config.yaml'dan yüklenen tam config dict'i
        """
        notif_cfg = config.get('notifications', {})
        tg_cfg = notif_cfg.get('telegram', {})

        self.enabled: bool = (
            notif_cfg.get('enabled', False) and
            tg_cfg.get('enabled', False)
        )
        self.bot_token: str = tg_cfg.get('bot_token', '')
        self.chat_id: str = str(tg_cfg.get('chat_id', ''))
        self.cooldown: float = float(tg_cfg.get('cooldown_per_track', 60))

        # Track başına son bildirim zamanı (spam önleme)
        self._last_sent: Dict[int, float] = {}
        self._lock = threading.Lock()

        if self.enabled:
            ok = self._validate_config()
            if ok:
                print("   ✅ Telegram Notifier aktif")
                print(f"   Chat ID : {self.chat_id}")
                print(f"   Cooldown: {self.cooldown}s / track")
            else:
                self.enabled = False
        else:
            print("   ℹ️  Telegram Notifier pasif (config.yaml → notifications.telegram.enabled)")

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def notify(
        self,
        original_frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        violation_data: dict,
    ) -> None:
        """
        İhlal bildirimi gönder — non-blocking.

        Args:
            original_frame : Tam çözünürlük frame (kırpma kalitesi için)
            bbox           : (x1, y1, x2, y2) — original_frame koordinatlarında
            violation_data : ViolationEngine'den dönen ihlal dict'i
        """
        if not self.enabled:
            return

        track_id: int = violation_data.get('track_id', -1)

        # Cooldown kontrolü (thread-safe)
        now = time.time()
        with self._lock:
            last = self._last_sent.get(track_id, 0.0)
            if now - last < self.cooldown:
                return
            self._last_sent[track_id] = now

        # Kırpılmış kişi görüntüsünü hazırla
        crop = self._crop_person(original_frame, bbox)
        if crop is None or crop.size == 0:
            # Bbox geçersizse tam frame'i gönder
            crop = original_frame.copy()

        # Arka planda gönder
        threading.Thread(
            target=self._send,
            args=(crop, violation_data),
            daemon=True,
        ).start()

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _crop_person(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        pad: float = 0.18,
    ) -> Optional[np.ndarray]:
        """
        Bbox'ı frame'den kırp, etrafına padding ekle.

        Args:
            pad: Bbox genişlik/yüksekliğinin bu oranı kadar padding
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        px = int(bw * pad)
        py = int(bh * pad)

        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)

        return frame[y1:y2, x1:x2].copy()

    def _send(self, crop: np.ndarray, violation_data: dict) -> None:
        """Telegram'a fotoğraf + caption gönder."""
        try:
            # JPEG'e encode et
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
            ok, buf = cv2.imencode('.jpg', crop, encode_params)
            if not ok:
                print("   ❌ Görüntü encode edilemedi")
                return

            img_bytes = io.BytesIO(buf.tobytes())

            # Caption metni
            track_id  = violation_data.get('track_id', '?')
            zone      = violation_data.get('zone_name', '?')
            ts        = violation_data.get('timestamp',
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            roi_time  = violation_data.get('time_in_roi', 0.0)

            caption = (
                f"🚨 *ALAN İHLALİ TESPİT EDİLDİ*\n\n"
                f"🆔 Track ID  : `#{track_id}`\n"
                f"📍 Bölge     : `{zone}`\n"
                f"⏰ Zaman     : `{ts}`\n"
                f"⏱️ Bölgede kalma: `{roi_time:.1f}s`"
            )

            url = self.API_URL.format(token=self.bot_token, method='sendPhoto')
            resp = requests.post(
                url,
                data={
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'Markdown',
                },
                files={'photo': ('violation.jpg', img_bytes, 'image/jpeg')},
                timeout=15,
            )

            if resp.status_code == 200:
                print(f"   📱 Telegram → Track #{track_id} bildirimi gönderildi")
            else:
                err = resp.json().get('description', resp.text)
                print(f"   ❌ Telegram HTTP {resp.status_code}: {err}")

        except requests.exceptions.Timeout:
            print("   ❌ Telegram zaman aşımı (15s)")
        except requests.exceptions.ConnectionError:
            print("   ❌ Telegram bağlantı hatası (internet var mı?)")
        except Exception as exc:
            print(f"   ❌ Telegram gönderim hatası: {exc}")

    def _validate_config(self) -> bool:
        """Token ve chat_id dolu mu kontrol et."""
        if not self.bot_token or self.bot_token == 'YOUR_BOT_TOKEN_HERE':
            print("   ❌ Telegram bot_token ayarlanmamış!")
            return False
        if not self.chat_id or self.chat_id == 'YOUR_CHAT_ID_HERE':
            print("   ❌ Telegram chat_id ayarlanmamış!")
            return False
        return True
