import os
import cv2
import numpy as np
import time
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tqdm import tqdm
import shutil

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
processing_details = {
    "frame_processed": 0,
    "total_frames": 0
}
processing_start_time = 0.0

# Load YOLOv8 Model (cần mô hình hỗ trợ "license_plate" nếu muốn phát hiện biển số)
model = YOLO("yolov8n.pt")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def process_video(input_path, output_path, report_path):
    global processing_progress, processing_details, processing_start_time
    processing_start_time = time.time()
    print("[INFO] Bắt đầu xử lý video giám sát tội phạm (dynamic frame size)...")
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    processing_details["total_frames"] = total_frames
    
    # Dùng để đếm đối tượng duy nhất (chỉ ghi nhận lần đầu phát hiện)
    unique_person_ids = set()
    first_appearance = {}  # first_appearance[track_id] = second (first time appearance)
    max_people_count = 0   # Số người tối đa xuất hiện đồng thời
    people_by_second = {}  # Số người xuất hiện theo từng giây
    
    # VideoWriter sẽ được khởi tạo sau khi đọc frame đầu tiên
    out = None

    with open(report_path, "w") as report:
        report.write("=== BÁO CÁO GIÁM SÁT ===\n")
    
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Hết video.")
                break
            
            # Kiểm tra khung hình hợp lệ
            if frame is None or frame.size == 0:
                print("[WARNING] Khung hình không hợp lệ, bỏ qua.")
                continue
            
            # Lấy kích thước khung hình động
            height, width, _ = frame.shape
            
            # Khởi tạo VideoWriter nếu chưa được khởi tạo
            if out is None:
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            
            # Resize khung hình để tăng tốc xử lý
            scale_factor = 0.3
            small_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
            
            # Chạy YOLOv8 để phát hiện đối tượng
            results = model.track(source=small_frame, persist=True, conf=0.5)
            second = frame_count // fps
            
            # Đếm số người trong frame hiện tại
            active_person_ids = set()
            
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    track_id = box.id[0] if box.id is not None else None
                    
                    if class_name == "person" and track_id is not None:
                        # Thêm vào danh sách active trong frame hiện tại
                        active_person_ids.add(track_id)
                        
                        # Nếu đây là lần đầu gặp, lưu lại thời điểm
                        if track_id not in unique_person_ids:
                            unique_person_ids.add(track_id)
                            first_appearance[track_id] = second
                        
                        # Vẽ bounding box trên khung hình gốc
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1 = int(x1 * (1 / scale_factor))
                        y1 = int(y1 * (1 / scale_factor))
                        x2 = int(x2 * (1 / scale_factor))
                        y2 = int(y2 * (1 / scale_factor))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {track_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Cập nhật số người tối đa xuất hiện đồng thời
            max_people_count = max(max_people_count, len(active_person_ids))
            
            # Ghi nhận số người xuất hiện theo từng giây
            if second not in people_by_second:
                people_by_second[second] = len(active_person_ids)
            
            # Ghi khung hình đã xử lý vào video đầu ra
            out.write(frame)
            frame_count += 1
            processing_progress = int((frame_count / total_frames) * 100)
    
        # Viết báo cáo rõ ràng và chính xác
        report.write(f"Tổng số frame: {total_frames}\n")
        report.write(f"Tổng số người phát hiện: {len(unique_person_ids)}\n")
        report.write(f"Số người tối đa xuất hiện đồng thời: {max_people_count}\n\n")
        
        # Ghi lại thời gian xuất hiện đầu tiên của mỗi track ID
        report.write("Thời gian xuất hiện của các đối tượng:\n")
        for track_id in unique_person_ids:
            first_seen = first_appearance.get(track_id, 0)
            formatted_time = f"{first_seen // 60:02d}:{first_seen % 60:02d}"  # Định dạng thời gian thành phút:giây
            report.write(f"  - Track ID: {track_id}, Xuất hiện lúc: {formatted_time}\n")
        
        # Ghi số người xuất hiện theo từng giây
        report.write("\nSố người xuất hiện theo từng giây:\n")
        for second, count in sorted(people_by_second.items()):
            report.write(f"  - {second} giây: {count} người\n")

    cap.release()
    if out:
        out.release()
    processing_progress = 100
    print("[INFO] Đã hoàn tất xử lý video giám sát tội phạm.")
    return []  # Hiện không capture ảnh, trả về danh sách rỗng

@app.route("/", methods=["GET"])
def upload_page():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global processing_progress, processing_filename

    # Xóa cache trước khi xử lý file mới
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, REPORT_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Xóa file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Xóa thư mục
            except Exception as e:
                print(f"[WARNING] Không thể xóa file {file_path}: {e}")

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
        print(f"[INFO] Nhận được file: {filename}")
        file.save(input_path)
        processing_filename = f"processed_{filename}"
        processing_progress = 0

        print("[INFO] Bắt đầu gọi hàm process_video...")
        captured_files = process_video(input_path, output_path, report_path)
        print("[INFO] Hoàn thành upload_file và process_video.")

        # Trả danh sách file ảnh
        return jsonify({
            "message": "Upload successful",
            "filename": filename,
            "captures": captured_files
        }), 200
    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500

@app.route("/processing-status")
def processing_status():
    elapsed_time = time.time() - processing_start_time
    frame_processed = processing_details["frame_processed"]
    total_frames = processing_details["total_frames"]
    
    # Tính thời gian ước tính còn lại (đơn giản: tỷ lệ khung hình đã xử lý so với thời gian)
    if frame_processed > 0:
        estimate_total = elapsed_time / frame_processed * total_frames
        estimate_remaining = max(0, estimate_total - elapsed_time)
    else:
        estimate_remaining = 0

    return jsonify({
        "progress": processing_progress,
        "filename": processing_filename,
        "frame_processed": frame_processed,
        "total_frames": total_frames,
        "elapsed_time": round(elapsed_time, 2),
        "remaining_time": round(estimate_remaining, 2)
    })
    
if __name__ == "__main__":
    app.run(debug=True)
