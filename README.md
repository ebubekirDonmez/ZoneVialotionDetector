# ZoneVialotionDetector

A real-time computer vision pipeline that detects unauthorized access to restricted zones and delivers instant alerts via Telegram.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

Most security cameras record everything but understand nothing. This system watches a live video feed, maintains persistent identity tracking across frames, and enforces configurable zone access rules — triggering a real-time Telegram alert with a cropped photo of the individual the moment a violation is confirmed.

---

## Pipeline

```
Video Input
    │
    ▼
┌─────────────┐
│   YOLOv8   │  ← Per-frame person detection
└──────┬──────┘
       │
    ▼
┌─────────────┐
│  DeepSORT  │  ← Multi-object tracking with Re-ID embeddings
└──────┬──────┘
       │
    ▼
┌──────────────────┐
│  ROI Checker    │  ← Polygon zone containment check
└──────┬───────────┘
       │
    ▼
┌──────────────────────┐
│  Violation Engine   │  ← Time-based rules + grace period
└──────┬───────────────┘
       │
    ▼
┌──────────────────────┐
│ Telegram Notifier   │  ← Async photo delivery
└──────────────────────┘
```

---

## Features

- **Interactive ROI Editor** — draw complex polygon zones directly on the video frame
- **Time-based access rules** — define authorized hours; entries outside that window trigger violations
- **1.5s grace period** — filters false exits caused by frame clipping or tracking jitter
- **Decoupled resolutions** — inference runs at reduced scale, display renders at native resolution
- **Async notifications** — Telegram delivery runs on a background thread with zero impact on the detection loop
- **Per-track cooldown** — prevents duplicate alerts for the same individual

---

## Project Structure

```
zone_violation_detection/
│
├── main.py                  # Main pipeline orchestrator
├── config.yaml              # All system settings
├── requirements.txt
│
├── modules/
│   ├── roi_checker.py       # Polygon containment logic
│   ├── violation_engine.py  # Rule engine + track state management
│   └── notifier.py          # Telegram Bot API integration
│
├── tools/
│   └── roi_editor.py        # Interactive zone drawing tool
│
├── data/                    # ROI zone JSON files (auto-generated)
├── outputs/
│   ├── events/              # Violation snapshots
│   └── videos/              # Processed output videos
└── logs/
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/zone_violation_detection.git
cd zone_violation_detection

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml` to set your YOLO model, detection thresholds, authorized hours, and Telegram credentials.

```yaml
violation_rules:
  min_time_in_roi: 3.0       # seconds before violation is confirmed
  time_based:
    enabled: true
    start_time: "08:00"      # authorized window start
    end_time: "16:00"        # authorized window end

notifications:
  telegram:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
```

**Getting your Telegram credentials:**
1. Message `@BotFather` → `/newbot` → copy the token
2. Send any message to your bot, then open:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Find `"chat": {"id": ...}` — that's your chat ID

### 3. Draw ROI zones

```bash
python main.py --video sample.mp4 --edit-roi
```

| Control | Action |
|---|---|
| Left click | Add point |
| Click near first point | Close polygon |
| `Enter` / `Space` | Save current zone |
| `U` | Undo last zone |
| `S` | Save and exit |
| `Q` / `Esc` | Exit without saving |

### 4. Run

```bash
python main.py --video sample.mp4
```

| Key | Action |
|---|---|
| `Q` | Quit |
| `T` | Toggle time-based rule on/off |

---

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `yolo.model` | `yolov8n.pt` | Model size (n/s/m/l/x) |
| `yolo.confidence` | `0.5` | Detection confidence threshold |
| `yolo.device` | `cuda` | `cuda` or `cpu` |
| `deepsort.max_age` | `30` | Frames to keep a lost track alive |
| `deepsort.n_init` | `3` | Detections required before ID is confirmed |
| `roi.check_method` | `bottom_center` | Point used for zone check (`bottom_center`, `center`, `any`, `all`) |
| `violation_rules.min_time_in_roi` | `3.0` | Seconds in zone before violation triggers |
| `video.resize_scale` | `0.5` | Inference resolution scale (0.5 = 50%) |
| `video.window_scale` | `1.0` | Display window scale relative to original |
| `notifications.telegram.cooldown_per_track` | `60` | Seconds between alerts for the same individual |

---

## Tech Stack

| Component | Library |
|---|---|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Multi-Object Tracking | [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Notifications | Telegram Bot API via `requests` |

---

## Security Note

Never commit real credentials to version control. Add `config.yaml` to `.gitignore` and use a separate local config file with your actual bot token and chat ID.

---

## License

MIT
