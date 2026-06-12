import os
import json
from flask import (Flask, render_template, request,
                   redirect, url_for, jsonify, send_file)
from datetime import datetime

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs("uploads",       exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

# ── Demo state (no YOLO needed for UI demo) ───────────────────────────────────
VEHICLE_TYPES = ["bicycle", "car", "motorcycle", "bus", "truck"]

demo_counts = {
    "bicycle":    {"IN": 3,  "OUT": 2},
    "car":        {"IN": 45, "OUT": 38},
    "motorcycle": {"IN": 12, "OUT": 10},
    "bus":        {"IN": 5,  "OUT": 4},
    "truck":      {"IN": 8,  "OUT": 6},
}

demo_logs = [
    {"time": "08:15:32", "vehicle": "car",        "direction": "IN",  "speed": 52.3},
    {"time": "08:15:45", "vehicle": "motorcycle", "direction": "IN",  "speed": 67.1},
    {"time": "08:16:02", "vehicle": "truck",      "direction": "OUT", "speed": 38.5},
    {"time": "08:16:18", "vehicle": "car",        "direction": "IN",  "speed": 48.9},
    {"time": "08:16:35", "vehicle": "bus",        "direction": "OUT", "speed": 32.4},
]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stop", methods=["POST"])
def stop():
    return jsonify({"status": "stopped"})


@app.route("/video_feed")
def video_feed():
    # On Render show a placeholder — no live video
    return "", 204


@app.route("/api/status")
def api_status():
    grand_total = sum(v["IN"] + v["OUT"] for v in demo_counts.values())
    return jsonify({
        "running":     False,
        "finished":    True,
        "error":       None,
        "congestion":  "MODERATE",
        "grand_total": grand_total,
        "avg_speed":   47.3,
        "counts":      demo_counts,
        "logs":        demo_logs,
        "demo_mode":   True,
    })


@app.route("/results")
def results():
    grand_total = sum(v["IN"] + v["OUT"] for v in demo_counts.values())
    return render_template("results.html",
                           counts=demo_counts,
                           grand_total=grand_total,
                           avg_speed=47.3,
                           logs=demo_logs)


@app.route("/download_csv")
def download_csv():
    # Generate a demo CSV if no real one exists
    csv_path = "output.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("Time,Vehicle,Direction,Speed(km/h),Confidence,Congestion\n")
            for log in demo_logs:
                f.write(f"{datetime.now().strftime('%Y-%m-%d')} {log['time']},"
                        f"{log['vehicle']},{log['direction']},"
                        f"{log['speed']},0.85,MODERATE\n")
    return send_file(csv_path,
                     as_attachment=True,
                     download_name="traffic_report.csv")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, threaded=True, host="0.0.0.0", port=port)
