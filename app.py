import os
import cv2
import csv
import time
import shutil
import threading
from collections import defaultdict
from datetime import datetime
from flask import (Flask, render_template, request,
                   redirect, url_for, Response,
                   jsonify, send_file)
from ultralytics import YOLO

from speed import estimate_speed
from plot import generate_graphs
from config import LINE_POSITION, OFFSET, FRAME_SKIP, PIXEL_TO_METER

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

os.makedirs("uploads",       exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

# ── Vehicle classes ───────────────────────────────────────────────────────────
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

BOX_COLORS = {
    "bicycle":    (255, 255,   0),
    "car":        (255,   0, 255),
    "motorcycle": (  0, 255, 255),
    "bus":        (  0, 255,   0),
    "truck":      (  0, 128, 255),
}

CONGESTION_LEVELS = [
    (5,   "FREE FLOW", (34, 197,  94)),
    (15,  "MODERATE",  (234, 179,  8)),
    (999, "CONGESTED", (239,  68, 68)),
]

# ── Global shared state ───────────────────────────────────────────────────────
state = {
    "running":     False,
    "finished":    False,
    "use_camera":  False,
    "counts":      {v: {"IN": 0, "OUT": 0} for v in VEHICLE_CLASSES.values()},
    "speed_log":   [],
    "traffic_log": [],
    "congestion":  "FREE FLOW",
    "frame":       None,
    "error":       None,
}
state_lock  = threading.Lock()

# ── Camera frame generator (separate thread) ──────────────────────────────────
camera_frame = None
camera_lock  = threading.Lock()
camera_thread_running = False


def get_congestion(count):
    for threshold, label, _ in CONGESTION_LEVELS:
        if count <= threshold:
            return label
    return "CONGESTED"


def reset_state():
    with state_lock:
        state["running"]     = False
        state["finished"]    = False
        state["counts"]      = {v: {"IN": 0, "OUT": 0} for v in VEHICLE_CLASSES.values()}
        state["speed_log"]   = []
        state["traffic_log"] = []
        state["congestion"]  = "FREE FLOW"
        state["frame"]       = None
        state["error"]       = None


def run_detection(video_path=None, use_camera=False):
    """Background detection thread — works for both video file and webcam."""
    global camera_frame

    try:
        model = YOLO("yolo26n.pt")

        # ── Open video source ─────────────────────────────────────────────────
        if use_camera:
            # Try camera indices 0, 1, 2 until one opens
            cap = None
            for idx in range(3):
                test = cv2.VideoCapture(idx)
                if test.isOpened():
                    cap = test
                    break
                test.release()
            if cap is None:
                with state_lock:
                    state["error"]   = "No camera found. Make sure webcam is connected."
                    state["running"] = False
                return
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                with state_lock:
                    state["error"]   = "Could not open video file."
                    state["running"] = False
                return

        # ── Set camera properties for better performance ──────────────────────
        if use_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

        track_history = defaultdict(list)
        counted_up    = set()
        counted_down  = set()
        frame_id      = 0

        csv_file = open("output.csv", "w", newline="")
        csv_w    = csv.writer(csv_file)
        csv_w.writerow(["Time", "Vehicle", "Direction",
                         "Speed(km/h)", "Confidence", "Congestion"])

        # For camera mode skip fewer frames so it feels more live
        skip = 2 if use_camera else FRAME_SKIP

        while True:
            with state_lock:
                if not state["running"]:
                    break

            ret, frame = cap.read()

            if not ret:
                if use_camera:
                    time.sleep(0.01)
                    continue   # keep trying for camera
                else:
                    break      # video finished

            frame_id += 1
            if frame_id % skip != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape

            results = model.track(
                frame, persist=True,
                conf=0.25, iou=0.45,
                tracker="bytetrack.yaml",
                verbose=False
            )

            # Counting line
            cv2.line(frame, (0, LINE_POSITION), (w, LINE_POSITION),
                     (220, 50, 50), 2)
            cv2.putText(frame, "COUNTING LINE",
                        (w // 2 - 65, LINE_POSITION - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 50, 50), 1)

            current_vehicles = 0

            if results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs   = results[0].boxes.conf.cpu().numpy()

                current_vehicles = sum(
                    1 for c in classes if int(c) in VEHICLE_CLASSES
                )

                for box, tid, cls, conf in zip(boxes, ids, classes, confs):
                    cls = int(cls)
                    if cls not in VEHICLE_CLASSES:
                        continue

                    label = VEHICLE_CLASSES[cls]
                    color = BOX_COLORS[label]
                    tid   = int(tid)

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    track_history[tid].append((cx, cy))
                    if len(track_history[tid]) > 30:
                        track_history[tid].pop(0)

                    for px, py in track_history[tid][-8:]:
                        cv2.circle(frame, (px, py), 2, color, -1)

                    speed      = estimate_speed(track_history[tid], tid)
                    congestion = get_congestion(current_vehicles)

                    if len(track_history[tid]) >= 2:
                        prev_y = track_history[tid][-2][1]
                        curr_y = track_history[tid][-1][1]

                        if prev_y < LINE_POSITION and curr_y >= LINE_POSITION:
                            if tid not in counted_down:
                                counted_down.add(tid)
                                with state_lock:
                                    state["counts"][label]["IN"] += 1
                                    if speed > 0:
                                        state["speed_log"].append(speed)
                                    state["traffic_log"].append({
                                        "time":      datetime.now().strftime("%H:%M:%S"),
                                        "vehicle":   label,
                                        "direction": "IN",
                                        "speed":     speed
                                    })
                                csv_w.writerow([
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    label, "IN", speed, f"{conf:.2f}", congestion
                                ])

                        elif prev_y > LINE_POSITION and curr_y <= LINE_POSITION:
                            if tid not in counted_up:
                                counted_up.add(tid)
                                with state_lock:
                                    state["counts"][label]["OUT"] += 1
                                    if speed > 0:
                                        state["speed_log"].append(speed)
                                    state["traffic_log"].append({
                                        "time":      datetime.now().strftime("%H:%M:%S"),
                                        "vehicle":   label,
                                        "direction": "OUT",
                                        "speed":     speed
                                    })
                                csv_w.writerow([
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    label, "OUT", speed, f"{conf:.2f}", congestion
                                ])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    txt = f"{label} {conf:.0%} | {speed}km/h"
                    tw  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
                    cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, txt, (x1 + 2, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

            with state_lock:
                state["congestion"] = get_congestion(current_vehicles)
                _, jpeg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 60])
                state["frame"] = jpeg.tobytes()

        cap.release()
        csv_file.close()

        # Generate and copy graphs (video mode only)
        if not use_camera:
            generate_graphs()
            for f in ["traffic_analysis_dashboard.png", "accuracy_report.png"]:
                src = f"graphs/{f}"
                dst = f"static/graphs/{f}"
                if os.path.exists(src):
                    shutil.copy(src, dst)

        with state_lock:
            state["running"]  = False
            state["finished"] = True

    except Exception as e:
        with state_lock:
            state["error"]   = str(e)
            state["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    reset_state()

    use_camera = request.form.get("use_camera") == "true"

    if use_camera:
        with state_lock:
            state["use_camera"] = True
            state["running"]    = True
        t = threading.Thread(
            target=run_detection, args=(None, True), daemon=True)
        t.start()
        return redirect(url_for("dashboard"))

    if "video" in request.files and request.files["video"].filename:
        video = request.files["video"]
        path  = os.path.join("uploads", "video.mp4")
        video.save(path)
        with state_lock:
            state["use_camera"] = False
            state["running"]    = True
        t = threading.Thread(
            target=run_detection, args=(path, False), daemon=True)
        t.start()
        return redirect(url_for("dashboard"))

    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/stop", methods=["POST"])
def stop():
    with state_lock:
        state["running"] = False
    return jsonify({"status": "stopping"})


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with state_lock:
                frame = state.get("frame")
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)   # ~25 fps max to browser
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state_lock:
        counts    = dict(state["counts"])
        speed_log = list(state["speed_log"])
        logs      = list(state["traffic_log"][-10:])
        running   = state["running"]
        finished  = state["finished"]
        error     = state["error"]
        congestion = state["congestion"]

    grand_total = sum(v["IN"] + v["OUT"] for v in counts.values())
    avg_speed   = (round(sum(speed_log[-50:]) / len(speed_log[-50:]), 1)
                   if speed_log else 0)

    return jsonify({
        "running":     running,
        "finished":    finished,
        "error":       error,
        "congestion":  congestion,
        "grand_total": grand_total,
        "avg_speed":   avg_speed,
        "counts":      counts,
        "logs":        logs,
    })


@app.route("/results")
def results():
    with state_lock:
        counts    = dict(state["counts"])
        speed_log = list(state["speed_log"])
        logs      = list(state["traffic_log"])

    grand_total = sum(v["IN"] + v["OUT"] for v in counts.values())
    avg_speed   = (round(sum(speed_log) / len(speed_log), 1)
                   if speed_log else 0)
    return render_template("results.html",
                           counts=counts,
                           grand_total=grand_total,
                           avg_speed=avg_speed,
                           logs=logs)


@app.route("/download_csv")
def download_csv():
    if os.path.exists("output.csv"):
        return send_file("output.csv",
                         as_attachment=True,
                         download_name="traffic_report.csv")
    return "No data yet", 404


if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)
