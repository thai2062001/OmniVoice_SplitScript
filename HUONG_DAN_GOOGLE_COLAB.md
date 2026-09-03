# 🚀 HƯỚNG DẪN CHẠY OMNIVOICE STUDIO TRÊN GOOGLE COLAB

Tài liệu này hướng dẫn chi tiết từng bước thiết lập và chạy giao diện OmniVoice AI Studio trên Google Colab (sử dụng GPU T4 miễn phí), đồng thời khắc phục triệt để lỗi kết nối Google Drive (`credential propagation was unsuccessful`).

---

## 📌 BƯỚC 1: Chọn GPU cho Colab (Bắt buộc)
1. Trên thanh menu Colab, chọn **Runtime (Thời gian chạy)** ➔ **Change runtime type (Thay đổi loại thời gian chạy)**.
2. Chọn **T4 GPU** ➔ Bấm **Save (Lưu)**.

---

## 📌 BƯỚC 2: Kết nối Google Drive & Tạo liên kết lưu giọng vĩnh viễn

> 💡 **LƯU Ý VỀ LỖI `credential propagation was unsuccessful`**:
> - Lỗi này xảy ra khi trình duyệt chặn popup ủy quyền hoặc phiên xác thực tài khoản Google bị lệch.
> - **Giải pháp tối ưu**: Dùng khối lệnh tự động nhận diện dưới đây. Nếu bạn **không muốn mount Drive**, code sẽ tự động tạo thư mục cục bộ mà không bị dừng chương trình.
> - **Cách bấm thủ công**: Bạn cũng có thể bấm vào biểu tượng **📁 Thư mục** ở cột bên trái của Colab ➔ bấm biểu tượng **Mount Drive (Gắn ổ đĩa)** và nhấn *Cho phép kết nối*.

Chạy Cell này trong Colab:

```python
# ==============================================================
# CELL 1: KẾT NỐI GOOGLE DRIVE (TỰ ĐỘNG XỬ LÝ LỖI ỦY QUYỀN)
# ==============================================================
import os

use_drive = False
try:
    from google.colab import auth
    # Xác thực trước để tránh lỗi credential propagation
    auth.authenticate_user()
    
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    use_drive = True
    print("✅ Đã kết nối Google Drive thành công!")
except Exception as e:
    print(f"⚠️ Không thể mount Drive tự động ({e}).")
    print("👉 Đang chuyển sang chế độ lưu trữ cục bộ tạm thời tại /content/saved_voices.")

# Tạo thư mục lưu hồ sơ giọng cố định (.pt)
if use_drive:
    SAVED_DIR = "/content/drive/MyDrive/OmniVoice_SavedVoices"
else:
    SAVED_DIR = "/content/saved_voices"

os.makedirs(SAVED_DIR, exist_ok=True)
print(f"📁 Thư mục lưu trữ giọng (.pt): {SAVED_DIR}")
```

---

## 📌 BƯỚC 3: Tải mã nguồn mới nhất & Liên kết thư mục giọng

Chạy Cell này:

```python
# ==============================================================
# CELL 2: CLONE MÃ NGUỒN & LIÊN KẾT THƯ MỤC LƯU GIỌNG
# ==============================================================
# Xóa thư mục cũ nếu có để luôn nhận code mới nhất
!rm -rf OmniVoice_SplitScript
!git clone https://github.com/thai2062001/OmniVoice_SplitScript.git
%cd OmniVoice_SplitScript

# Tạo symlink thư mục saved_voices trỏ tới Drive (hoặc thư mục đã chọn)
!rm -rf omnivoice/cli/saved_voices
!ln -s "{SAVED_DIR}" omnivoice/cli/saved_voices

print("✅ Đã clone repo và liên kết thư mục saved_voices thành công!")
```

---

## 📌 BƯỚC 4: Cài đặt thư viện phụ thuộc

Chạy Cell này:

```python
# ==============================================================
# CELL 3: CÀI ĐẶT THƯ VIỆN CẦN THIẾT
# ==============================================================
!pip install -e .
!pip install gradio soundfile torchaudio librosa
!apt-get install -y ffmpeg
```

---

## 📌 BƯỚC 5: Khởi chạy OmniVoice AI Studio với đường link Public (Gradio Share)

Chạy Cell này để mở giao diện Web đầy đủ 6 Tabs:

```python
# ==============================================================
# CELL 4: KHỞI CHẠY GIAO DIỆN WEB STUDIO (PUBLIC SHARE LINK)
# ==============================================================
# Lệnh này sẽ tải mô hình k2-fsa/OmniVoice và tạo link công khai gradio.live
!python -m omnivoice.cli.demo --model k2-fsa/OmniVoice --share
```

Khi chạy xong, Colab sẽ hiển thị một đường link dạng:
👉 **`Running on public URL: https://xxxx.gradio.live`**

Bạn chỉ cần click vào link đó để mở giao diện web và sử dụng đầy đủ các tính năng:
1. 🎙️ **Voice Manager**: Tạo và lưu vĩnh viễn hồ sơ giọng (`.pt`) trực tiếp vào Google Drive của bạn.
2. 🗣️ **Voice Clone**: Chọn giọng từ Drive để sinh giọng ngay không cần upload lại.
3. 📦 **Batch Voice Clone**: Sinh đồng thời 5 câu thoại.
4. 📜 **Script Clone**: Phân tích cảm xúc kịch bản bằng Gemini AI và sinh giọng timeline phân đoạn.
5. 🎨 **Voice Design**: Thiết kế giọng nói theo đặc tính nhân vật.
6. 🧩 **Ghép Audio**: Ghép nối tự nhiên theo số thứ tự file wav hoàn chỉnh.
