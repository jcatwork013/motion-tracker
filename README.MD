# 🏆 Motion Tracker - Nhận diện con người trong video bằng YOLOv8

Dự án này sử dụng **YOLOv8 (`ultralytics`)** để **nhận diện con người trong video**, vẽ khung nhận diện và lưu kết quả thành một video mới. Ngoài ra, hệ thống tạo **báo cáo thời gian xuất hiện của con người** trong video.

---

## 🚀 **1. Yêu cầu hệ thống**
### **1.1. Phiên bản Python**
- ✅ **Python 3.8 - 3.11** (Khuyến nghị dùng **Python 3.11**)

📌 Kiểm tra phiên bản Python:
```bash
python --version

python3 -m venv venv
source venv/bin/activate  # (Trên macOS/Linux)
venv\Scripts\activate  # (Trên Windows)


Yêu cầu thư viện & phiên bản
📦 Thư viện	🏷 Phiên bản yêu cầu
ultralytics	>=8.0.0
torch	>=1.8.0, <2.0.0
flask	>=2.0.0
opencv-python	>=4.5.0
numpy	1.26.4
tqdm	>=4.64.0
