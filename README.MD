# 🏆 Motion Tracker - Nhận diện con người trong video bằng YOLOv8

Dự án này sử dụng **YOLOv8 (`ultralytics`)** để **nhận diện con người trong video**, vẽ khung nhận diện và lưu kết quả thành một video mới. Ngoài ra, hệ thống tạo **báo cáo thời gian xuất hiện của con người** trong video.

---

## 🚀 **1. Yêu cầu hệ thống**
### **1.1. Phiên bản Python**
- ✅ **Python 3.8 - 3.11** (Khuyến nghị dùng **Python 3.11**)

📌 Kiểm tra phiên bản Python:
python --version

📌 Nếu Python chưa đúng phiên bản, tải tại: https://www.python.org/downloads/

1.2. Yêu cầu thư viện & phiên bản
📦 Thư viện	🏷 Phiên bản yêu cầu
ultralytics	>=8.0.0
torch	>=1.8.0, <2.0.0 (Chạy CPU, tránh lỗi)
flask	>=2.0.0
opencv-python	>=4.5.0
numpy	1.26.4 (Tránh lỗi torch không tương thích)
tqdm	>=4.64.0
📌 Cài đặt đúng torch để chạy trên CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

🛠 2. Cài đặt môi trường
2.1. Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate  # (Trên macOS/Linux)
venv\Scripts\activate  # (Trên Windows)

2.2. Cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install ultralytics flask opencv-python numpy==1.26.4 tqdm

📂 3. Cấu trúc thư mục

motion-tracker/
│── app.py                 # Flask server xử lý video
│── venv/                  # Môi trường ảo (được tạo sau khi chạy)
│── static/
│   ├── processed/         # Video đã xử lý
│── uploads/               # Video tải lên
│── reports/               # Báo cáo thời gian có người xuất hiện
│── templates/
│   ├── upload.html        # Giao diện web
│── README.md              # Hướng dẫn cài đặt


📥 4. Tải xuống mô hình YOLOv8
Trước khi chạy, cần tải mô hình YOLOv8 về:


yolo task=detect mode=predict model=yolov8n.pt source=https://ultralytics.com/images/bus.jpg

Sau đó, kiểm tra tệp yolov8n.pt đã tải về trong thư mục dự án.

🎯 5. Chạy ứng dụng
Sau khi cài đặt thư viện, chạy ứng dụng bằng lệnh:

python app.py

📌 Nếu bị lỗi OSError: [Errno 48] Address already in use, hãy chạy:

lsof -i :5000  # Tìm tiến trình đang dùng cổng 5000
kill -9 <PID>  # Thay <PID> bằng số PID thực tế

Hoặc đổi cổng trong app.py:

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

🌍 6. Truy cập giao diện web
Sau khi chạy ứng dụng, mở trình duyệt và truy cập:

http://127.0.0.1:5000/
📌 Tính năng trên UI:

📤 Upload video (Hỗ trợ .mp4, .avi, .mov).
📊 Theo dõi tiến trình xử lý video.
📥 Tải xuống video đã xử lý và báo cáo nhận diện (motion_report.txt).
