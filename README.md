# Rugby Highlight Annotation & Training System

A full-stack tool for annotating rugby match video clips and training a 3D-CNN (R3D-18) model to detect and score highlight events.

---

## System Overview

```
frontend (React + Vite)   →   backend (FastAPI)   →   trainer.py (PyTorch R3D-18)
      :5173                        :8000                   subprocess
```

**Workflow:**

1. Register a match video file
2. Extract sliding-window clips (8s / 16s / 32s)
3. Annotate each clip with an event class and highlight score
4. Start training — the model trains in the background
5. Export labeled data as CSV or JSON

---

## Prerequisites

| Tool            | Version                |
| --------------- | ---------------------- |
| Python          | 3.11+                  |
| Node.js         | 18+                    |
| CUDA (optional) | 11.8+ for GPU training |

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd 3D_CNN_Video_Model
```

### 2. Set up the Python backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download FFmpeg (Windows only)

FFmpeg is required for video clip extraction. Run this once:

```bash
python download_ffmpeg.py
```

This downloads `ffmpeg.exe` and `ffprobe.exe` into `backend/bin/`.

> On macOS/Linux, install FFmpeg via your package manager (`brew install ffmpeg` or `apt install ffmpeg`) and make sure it is on your `PATH`.

### 4. Set up the React frontend

```bash
cd ../frontend
npm install
```

---

## Running the System

You need **two terminals** running at the same time.

### Terminal 1 — Start the backend

```bash
cd backend
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive API docs: `http://localhost:8000/docs`

### Terminal 2 — Start the frontend

```bash
cd frontend
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## Using the Application

### Step 1: Register a Match

- Go to the **Matches** page
- Enter a match name and the **full absolute path** to your video file (e.g. `C:\Videos\match1.mp4`)
- Click **Add Match**

### Step 2: Extract Clips

- Click **Extract Clips** next to the match
- The backend will generate overlapping sliding-window clips at 8s, 16s, and 32s window sizes
- Watch the extraction progress bar until it completes

### Step 3: Annotate Clips

- Go to the **Annotate** page
- Select a match and (optionally) a window size filter
- For each clip:
  - Watch the video
  - Select the **event class** (try, tackle, lineout, etc.)
  - Set the **highlight score** (0.0 = not a highlight, 1.0 = key moment)
  - Adjust start/end times if needed
  - Click **Save Label** or **Skip**

### Step 4: Train the Model

- Go to the **Training** page
- Click **Start Training**
- Training runs as a background process — the page polls every 3 seconds and shows:
  - Current epoch (out of 40)
  - Train loss / Val loss / Val accuracy
  - Live loss curve chart
  - Raw training log output
- Click **Stop** to interrupt training (a checkpoint is saved automatically)

**Training phases:**
| Epochs | Backbone | Learning Rate |
|--------|----------|---------------|
| 1–10 | Frozen (heads only) | 1e-3 |
| 11–30 | Partial (layer3 + layer4) | 1e-4 |
| 31–40 | Full fine-tune | 1e-5 |

Saved model checkpoints:

- `~/highlight_system/models/best_model.pt` — lowest validation loss
- `~/highlight_system/models/last_model.pt` — latest epoch

### Step 5: Export Labels

- Go to the **Export** page
- Download all annotations as **CSV** or **JSON**
- View per-class counts and highlight score distribution

---

## Data Storage

All data is stored under `~/highlight_system/` by default:

```
~/highlight_system/
├── clips/          # Extracted video clips
├── models/         # Trained model checkpoints
├── exports/        # Exported label files
├── labels.db       # SQLite database
└── training.log    # Live training output
```

Override the base directory by setting the environment variable:

```bash
# Windows PowerShell
$env:APP_BASE_DIR = "D:\my_data\highlight_system"

# macOS/Linux
export APP_BASE_DIR=/data/highlight_system
```

---

## Event Classes

The model classifies 23 rugby event types:

| Scoring      | Set Pieces | Play       | Discipline  | Context     |
| ------------ | ---------- | ---------- | ----------- | ----------- |
| try          | scrum      | tackle     | red_card    | replay      |
| conversion   | lineout    | ruck       | yellow_card | tmo_review  |
| penalty_kick |            | maul       | penalty     | normal_play |
| drop_goal    |            | intercept  | knock_on    |             |
| near_try     |            | line_break | turnover    |             |
|              |            | touch_kick |             |             |
|              |            | cross_kick |             |             |

---

## GPU vs CPU Training

The trainer automatically uses CUDA if available, otherwise falls back to CPU:

```
# GPU detected  → Training on cuda
# No GPU        → Training on cpu  (slower, ~10x)
```

To force CPU training, edit the start command in `backend/main.py` line ~308:

```python
"--device", "cpu",   # change from "cuda"
```

---

## Troubleshooting

**Backend fails to start**

- Make sure the virtual environment is activated before running `uvicorn`
- Check that port 8000 is not in use: `netstat -ano | findstr :8000`

**Extraction stuck or no clips appear**

- Verify FFmpeg is downloaded: check `backend/bin/ffmpeg.exe` exists
- Check the extraction log in the UI (click the log icon next to the match)

**Training won't start**

- You need at least one labeled clip before training can begin
- Check `~/highlight_system/training.log` for error messages

**Frontend shows "Network Error"**

- Make sure the backend is running on port 8000
- Check the browser console for CORS errors

**Out of memory during training**

- Reduce batch size in `backend/main.py` line ~308: `"--batch_size", "2"`
