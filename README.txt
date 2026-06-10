# ── HOW TO RUN THE WEB APP ────────────────────────────────────────────────────

## Step 1 — Install Flask
pip install flask

## Step 2 — Folder structure
Make sure your folder looks like this:

SAMTECH CODES/
├── app.py          ← Flask web app (new)
├── main.py         ← Your detection code
├── speed.py        ← Speed estimation
├── config.py       ← Settings
├── plot.py         ← Graphs
├── yolo26n.pt      ← YOLO model
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   └── results.html
├── static/
│   └── graphs/     ← Auto-created
└── uploads/        ← Auto-created

## Step 3 — Run the app
python app.py

## Step 4 — Open browser
Go to: http://localhost:5000

## Controls
- Upload video → processes and shows live counts
- Live camera → uses webcam for real-time detection
- Stop button → stops processing
- View Results → shows graphs and full report
- Download CSV → downloads traffic_report.csv
