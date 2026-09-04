import gradio as gr


def build_guide_tab():
    """Constructs the interactive User Guide & Documentation Tab in Vietnamese."""
    with gr.TabItem("📖 Hướng Dẫn Sử Dụng", id="tab_guide"):
        gr.Markdown(
            """
# 📚 HƯỚNG DẪN SỬ DỤNG OMNIVOICE AI STUDIO
Chào mừng bạn đến với **OmniVoice AI Studio** - Nền tảng lồng tiếng kịch bản, clone voice và thiết kế giọng nói AI đa ngôn ngữ (hỗ trợ hơn 600+ ngôn ngữ trên thế giới).
            """
        )

        with gr.Tabs():
            with gr.TabItem("1. 🎙️ Quản Lý Hồ Sơ Giọng"):
                gr.Markdown(
                    """
### 🎙️ Tab: Quản lý Hồ sơ Giọng Mẫu (Voice Manager)
**Mục đích:** Trích xuất đặc trưng giọng nói (voice embedding vector) từ file audio mẫu và lưu thành file `.pt` cố định để dùng lại nhiều lần mà không cần upload lại.

#### 📝 Các bước thực hiện:
1. **Chuẩn bị audio mẫu:** Chọn đoạn ghi âm giọng nói từ **3 đến 10 giây**, âm thanh rõ ràng, không lẫn nhạc nền hoặc tạp âm quá lớn.
2. **Nhập Tên hồ sơ:** Đặt tên nhận diện dễ nhớ (Ví dụ: `Nam_Ke_Chuyen`, `Nu_Truyen_Cam`, `MC_Vui_Ve`...).
3. **Văn bản tham chiếu (Tùy chọn):**
   - Nếu bạn biết chính xác câu người trong audio nói, hãy gõ vào để tăng độ chính xác.
   - Nếu để trống, hệ thống sẽ tự động dùng Whisper AI nhận diện hoặc gán tham chiếu mặc định.
4. **Bấm nút:** `⚡ Trích xuất & Lưu Hồ Sơ Giọng (.pt)`.
5. **Kiểm tra & Dùng thử:** Chọn tên giọng vừa lưu ở danh sách bên phải, bấm `▶ Sinh giọng đọc thử` để kiểm tra chất lượng giọng nói ngay lập tức.

> 💡 **Mẹo:** Các hồ sơ đã lưu ở đây sẽ tự động hiển thị trong tất cả các tab khác (`Sinh giọng kịch bản`, `Nhân bản giọng`, `Nhân bản hàng loạt`).
                    """
                )

            with gr.TabItem("2. 🎬 Sinh Giọng Theo Kịch Bản"):
                gr.Markdown(
                    """
### 🎬 Tab: Sinh Giọng Theo Kịch Bản (Script Clone)
**Mục đích:** Tự động tạo giọng đọc cho toàn bộ kịch bản video dài, lồng tiếng phim, video TikTok/YouTube với từng phân đoạn cảm xúc riêng biệt.

#### 📝 Các bước thực hiện:
1. **Chọn Giọng mẫu:**
   - Chọn `Sử dụng Hồ sơ giọng có sẵn` (chọn từ danh sách đã tạo ở Voice Manager).
   - Hoặc chọn `Tải lên Audio mẫu mới` nếu muốn dùng trực tiếp file âm thanh rời.
2. **Nhập Kịch bản:**
   - **Cách 1 (Kịch bản thô):** Dán văn bản bình thường, mỗi dòng là một câu phân đoạn.
   - **Cách 2 (Import file):** Bấm vào *Nhập kịch bản từ tệp tin* để tải file `.txt` hoặc `.md`.
3. **Gắn Cảm xúc & Hướng dẫn AI tự động:**
   - Bấm `✨ Phân tích kịch bản & Gợi ý cảm xúc (Gemini AI)` để AI tự động chia mốc thời gian, gắn cảm xúc (Hài hước, Kịch tính, Thì thầm...) phù hợp ngữ cảnh từng câu.
   - Bấm `✅ Đồng ý & Áp dụng` để cập nhật vào kịch bản chính.
4. **Tạo giọng đọc:**
   - Bấm `▶ Sinh đợt này (10 phân đoạn)` để tạo 10 câu đầu tiên.
   - Bấm `⏭ Đợt tiếp theo` để làm tiếp các câu sau.
   - Hoặc bấm `⚡ Sinh TOÀN BỘ kịch bản` để hệ thống tự động chạy hết toàn bộ.
5. **Nghe thử & Tải về:**
   - Mỗi câu có audio player riêng và nút `🔄 Thử lại` nếu câu nào bạn chưa ưng ý.
   - File `.zip` tổng hợp toàn bộ các câu `.wav` sẽ tự động cập nhật để bạn tải về máy một lần.

> 💡 **Tính năng nâng cao:**
> - **Tiếp tục tiến trình (Resume):** Nếu gặp sự cố ngắt quãng, chỉ cần bấm sinh lại, hệ thống sẽ tự động bỏ qua các câu đã tạo thành công trước đó để tiết kiệm tài nguyên GPU.
> - **Chế độ nháp (Turbo Draft):** Giảm bước tính toán xuống 16 steps để nghe thử kịch bản siêu tốc trước khi render chất lượng cao (24 steps).
                    """
                )

            with gr.TabItem("3. 👥 Nhân Bản Giọng Nói"):
                gr.Markdown(
                    """
### 👥 Tab: Nhân Bản Giọng Đơn & Hàng Loạt (Voice Clone / Batch Clone)
**Mục đích:** Clone giọng nhanh cho các câu đơn lẻ hoặc danh sách 5 câu độc lập.

#### 📝 Các bước thực hiện:
1. Chọn nguồn giọng: Hồ sơ `.pt` có sẵn hoặc tải lên file âm thanh mẫu.
2. Nhập văn bản cần đọc.
3. Chọn ngôn ngữ mong muốn (hỗ trợ hơn 600+ ngôn ngữ, hoặc để `Auto`).
4. *(Tùy chọn)* Nhập thêm **Chỉ dẫn phong cách (Instruct):** Ví dụ: `high pitch`, `low pitch`, `whisper`...
5. Bấm `🚀 Bắt Đầu Tạo Giọng` và nghe kết quả.
                    """
                )

            with gr.TabItem("4. 🎨 Thiết Kế Giọng Nói"):
                gr.Markdown(
                    """
### 🎨 Tab: Thiết Kế Giọng Nói AI (Voice Design)
**Mục đích:** Tạo ra một giọng đọc nhân tạo hoàn toàn mới từ số 0 bằng cách kết hợp các thuộc tính đặc trưng mà **không cần audio mẫu**.

#### 📝 Các thuộc tính có thể tùy biến:
- **Giới tính (Gender):** Nam (Male) / Nữ (Female)
- **Độ tuổi (Age):** Trẻ em (Child), Thiếu niên (Teenager), Thanh niên (Young Adult), Trung niên (Middle-aged), Lớn tuổi (Elderly).
- **Âm điệu (Pitch):** Cực trầm (Very Low), Trầm (Low), Vừa phải (Moderate), Cao (High), Cực cao (Very High).
- **Phong cách (Style):** Thì thầm (Whisper).
- **Chất giọng theo vùng miền / Quốc gia:** Anh (British), Mỹ (American), Nhật (Japanese), Trung Quốc (Chinese)...

> 💡 **Mẹo:** Sau khi tạo được giọng ưng ý ở tab này, bạn có thể lưu file audio vừa tạo sang tab **Quản lý Hồ sơ Giọng** để sử dụng vĩnh viễn cho kịch bản!
                    """
                )

            with gr.TabItem("5. 🧩 Ghép Audio Hoàn Chỉnh"):
                gr.Markdown(
                    """
### 🧩 Tab: Ghép Audio (Audio Merger)
**Mục đích:** Nối tự động các file phân đoạn audio nhỏ thành một file âm thanh hoàn chỉnh dài từ đầu đến cuối.

#### 📝 Các bước thực hiện:
1. **Chọn Chế độ:**
   - `Quét thư mục cục bộ`: Nhập đường dẫn thư mục (ví dụ: `audio/no_internet` hoặc thư mục cache kịch bản).
   - `Upload file trực tiếp`: Kéo thả danh sách các file audio cần ghép vào ô tải lên.
2. **Khoảng lặng giữa các câu (Gap):** Điều chỉnh khoảng im lặng (mặc định 0.3s) giữa các câu để giọng đọc tự nhiên, nhịp nhàng.
3. **Bấm:** `🚀 Bắt Đầu Ghép Audio`.
4. Hệ thống sẽ tự động nhận diện số thứ tự trong tên file (ví dụ `segment_1.wav`, `segment_2.wav`... `segment_10.wav`) để ghép theo thứ tự chính xác 100%, không bị xáo trộn.
                    """
                )

            with gr.TabItem("6. ⚡ Tối Ưu Cho Google Colab"):
                gr.Markdown(
                    """
### ⚡ Kinh Nghiệm Chạy Mượt Mà Trên Google Colab
1. **Lưu trữ vĩnh viễn trên Drive:** Luôn đảm bảo bạn đã chạy lệnh mount Google Drive. Mọi giọng nói lưu và các file kết quả sẽ được giữ an toàn tại `/content/drive/MyDrive/OmniVoice_Studio/`.
2. **Sử dụng Cloudflare Tunnel:** Sử dụng tham số `--tunnel cloudflare` để nhận đường link truy cập nhanh hơn và không bị ngắt kết nối giữa chừng khi tải file zip lớn.
3. **Chống ngắt kết nối (Keep-Alive):** Giữ tab trình duyệt mở hoặc dùng script Keep-Alive đã tích hợp sẵn trong notebook.
                    """
                )

    return {}
