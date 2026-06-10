import cv2
import csv
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO

from config import *
from speed import estimate_speed
from plot import generate_graphs

# ─── UNIVERSITY / PROJECT INFO (change to yours) ──────────────────────────────
UNIVERSITY  = "YOUR UNIVERSITY NAME"
PROJECT     = "Vehicle Detection & Counting System"
STUDENT     = "YOUR NAME"

# ─── VEHICLE CLASS IDs (YOLO26 / COCO) ────────────────────────────────────────
vehicle_classes = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ─── COLOR PER VEHICLE TYPE (BGR) ─────────────────────────────────────────────
BOX_COLORS = {
    "bicycle":    (255, 255,   0),
    "car":        (255,   0, 255),
    "motorcycle": (  0, 255, 255),
    "bus":        (  0, 255,   0),
    "truck":      (  0, 128, 255),
}

# ─── CONGESTION THRESHOLDS ────────────────────────────────────────────────────
CONGESTION_LEVELS = [
    (5,  "FREE FLOW",  (0, 255,   0)),
    (15, "MODERATE",   (0, 255, 255)),
    (999,"CONGESTED",  (0,   0, 255)),
]

def get_congestion(vehicle_count):
    for threshold, label, color in CONGESTION_LEVELS:
        if vehicle_count <= threshold:
            return label, color
    return "CONGESTED", (0, 0, 255)


def get_peak_minute(traffic_log):
    if not traffic_log:
        return "N/A"
    minute_counts = defaultdict(int)
    for entry in traffic_log:
        minute = entry["time"].strftime("%H:%M")
        minute_counts[minute] += 1
    peak = max(minute_counts, key=minute_counts.get)
    return f"{peak} ({minute_counts[peak]} vehicles)"


def print_summary(counts, traffic_log, avg_speeds):
    grand_total = sum(v["IN"] + v["OUT"] for v in counts.values())
    print("\n" + "="*55)
    print("       TRAFFIC ANALYSIS SUMMARY REPORT")
    print("="*55)
    print(f"  Total Vehicles Detected : {grand_total}")
    for v_label, v_counts in counts.items():
        total = v_counts["IN"] + v_counts["OUT"]
        pct   = (total / grand_total * 100) if grand_total > 0 else 0
        print(f"  {v_label:<12} IN={v_counts['IN']:>3}  OUT={v_counts['OUT']:>3}  TOTAL={total:>3}  ({pct:.1f}%)")
    print("-"*55)
    if avg_speeds:
        overall_avg = sum(avg_speeds) / len(avg_speeds)
        print(f"  Average Speed           : {overall_avg:.1f} km/h")
    peak = get_peak_minute(traffic_log)
    print(f"  Peak Traffic Minute     : {peak}")
    print("="*55 + "\n")


def finish_system():
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print_summary(counts, traffic_log, speed_log)
    print("Generating graphs...")
    generate_graphs()
    print("Done! Check the 'graphs/' folder.")


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
model = YOLO("yolo26n.pt")

# ─── LOAD VIDEO ───────────────────────────────────────────────────────────────
cap   = cv2.VideoCapture("video.mp4")
FPS   = cap.get(cv2.CAP_PROP_FPS) or 30

# ─── VIDEO WRITER ─────────────────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter("DEMO.mp4", fourcc, FPS, (640, 480))

# ─── DATA STORAGE ─────────────────────────────────────────────────────────────
track_history  = defaultdict(list)
counted_up     = set()
counted_down   = set()
counts         = {v: {"OUT": 0, "IN": 0} for v in vehicle_classes.values()}
traffic_log    = []
speed_log      = []
active_ids     = set()

# ─── CSV OUTPUT ───────────────────────────────────────────────────────────────
csv_file   = open("output.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Time", "Vehicle", "Direction", "Speed(km/h)", "Confidence", "Congestion_Level"])

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
frame_id = 0
paused   = False

print("Controls:  P=Pause  S=Screenshot  R=Reset  Q/ESC=Quit")

while True:

    if not paused:
        ret, frame = cap.read()

        if not ret:
            print("\nVideo finished.")
            finish_system()
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape

        results = model.track(
            frame,
            persist=True,
            conf=0.25,
            iou=0.45,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # Counting line
        cv2.line(frame, (0, LINE_POSITION), (w, LINE_POSITION), (0, 0, 255), 2)
        cv2.putText(frame, "COUNTING LINE",
                    (w // 2 - 60, LINE_POSITION - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        active_ids.clear()
        current_frame_vehicles = 0

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs   = results[0].boxes.conf.cpu().numpy()

            current_frame_vehicles = sum(1 for c in classes if int(c) in vehicle_classes)

            for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
                cls = int(cls)
                if cls not in vehicle_classes:
                    continue

                label    = vehicle_classes[cls]
                color    = BOX_COLORS[label]
                track_id = int(track_id)
                active_ids.add(track_id)

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                for i, (px, py) in enumerate(track_history[track_id][-8:]):
                    cv2.circle(frame, (px, py), 2, color, -1)

                speed = estimate_speed(track_history[track_id], track_id)
                if speed > 0:
                    speed_log.append(speed)

                congestion_label, _ = get_congestion(current_frame_vehicles)

                if len(track_history[track_id]) >= 2:
                    prev_y = track_history[track_id][-2][1]
                    curr_y = track_history[track_id][-1][1]

                    if prev_y < LINE_POSITION and curr_y >= LINE_POSITION:
                        if track_id not in counted_down:
                            counted_down.add(track_id)
                            counts[label]["IN"] += 1
                            now = datetime.now()
                            traffic_log.append({"time": now, "vehicle": label})
                            csv_writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"),
                                                 label, "IN", speed, f"{conf:.2f}", congestion_label])

                    elif prev_y > LINE_POSITION and curr_y <= LINE_POSITION:
                        if track_id not in counted_up:
                            counted_up.add(track_id)
                            counts[label]["OUT"] += 1
                            now = datetime.now()
                            traffic_log.append({"time": now, "vehicle": label})
                            csv_writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"),
                                                 label, "OUT", speed, f"{conf:.2f}", congestion_label])

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                display_text = f"{label} {conf:.0%} | ID:{track_id} | {speed}km/h"
                text_size    = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                cv2.rectangle(frame, (x1, y1 - 18), (x1 + text_size[0] + 4, y1), color, -1)
                cv2.putText(frame, display_text, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

                cv2.circle(frame, (cx, cy), 4, color, -1)

        # Congestion indicator
        congestion_label, congestion_color = get_congestion(current_frame_vehicles)
        cv2.rectangle(frame, (w - 175, 5), (w - 5, 32), (0, 0, 0), -1)
        cv2.putText(frame, f"TRAFFIC: {congestion_label}",
                    (w - 170, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, congestion_color, 2)

        # Count panel
        panel_x = 10
        panel_y = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (330, 22 + len(counts) * 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for v_label, v_counts in counts.items():
            color = BOX_COLORS[v_label]
            total = v_counts["IN"] + v_counts["OUT"]
            count_text = (f"{v_label:<10} IN={v_counts['IN']:>3}  "
                          f"OUT={v_counts['OUT']:>3}  TOTAL={total:>3}")
            cv2.putText(frame, count_text, (panel_x, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            panel_y += 30

        # Grand total bar
        grand_total = sum(v["IN"] + v["OUT"] for v in counts.values())
        cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"TOTAL VEHICLES COUNTED: {grand_total}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

        # Average speed
        if speed_log:
            avg_spd = sum(speed_log[-50:]) / len(speed_log[-50:])
            cv2.putText(frame, f"AVG SPEED: {avg_spd:.1f} km/h",
                        (w - 220, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)

        # University watermark
        cv2.putText(frame, UNIVERSITY,
                    (w // 2 - len(UNIVERSITY) * 4, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Paused overlay
    if paused:
        cv2.putText(frame, "PAUSED",
                    (w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.imshow("Nigerian Traffic Detection System", frame)
    out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q"):
        print("User quit.")
        finish_system()
        break
    elif key == ord("p"):
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
    elif key == ord("s"):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"screenshot_{ts}.jpg"
        cv2.imwrite(name, frame)
        print(f"Screenshot saved → {name}")
    elif key == ord("r"):
        for v in counts:
            counts[v] = {"IN": 0, "OUT": 0}
        counted_up.clear()
        counted_down.clear()
        traffic_log.clear()
        speed_log.clear()
        print("Counts reset!")

# Cleanup
try:
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
except Exception:
    pass
