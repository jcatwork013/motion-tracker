import os
import cv2
import numpy as np
import time
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tqdm import tqdm

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/processed"
REPORT_FOLDER = "reports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"mp4", "avi", "mov"}
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

processing_progress = 0
processing_filename = ""

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def process_video(input_path, output_path, report_path):
    global processing_progress
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motion_times = []

    with open(report_path, "w") as report:
        report.write("Time (seconds) | People Count\n")
        report.write("-------------------------\n")

        for _ in tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Nhận diện người bằng YOLOv8
            results = model(frame)

            people_count = 0
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])  # ID của object
                    if model.names[class_id] == "person":
                        people_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Ghi nhận thời điểm có người xuất hiện
            if people_count > 0:
                second = frame_count // fps
                if second not in motion_times:
                    motion_times.append(second)
                    report.write(f"{second} sec | {people_count} people\n")

            out.write(frame)
            frame_count += 1
            processing_progress = int((frame_count / total_frames) * 100)

    cap.release()
    out.release()
    processing_progress = 100


@app.route("/", methods=["GET"])
def upload_page():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global processing_progress, processing_filename

    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "Invalid file type", 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
    report_path = os.path.join(REPORT_FOLDER, f"report_{filename}.txt")

    try:
        file.save(input_path)
        processing_filename = f"processed_{filename}"
        processing_progress = 0

        # Bắt đầu xử lý video
        process_video(input_path, output_path, report_path)

        return jsonify({"message": "Upload successful", "filename": filename}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/processing-status")
def processing_status():
    return jsonify({"progress": processing_progress, "filename": processing_filename})
    
if __name__ == "__main__":
    app.run(debug=True)
