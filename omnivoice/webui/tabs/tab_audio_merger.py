import gradio as gr
from omnivoice.webui.audio_engine import process_audio_merger


def build_audio_merger_tab():
    """Constructs the Audio Merger Tab UI and internal event listeners."""
    with gr.TabItem("🧩 Ghép Audio (Audio Merger)"):
        gr.Markdown(
            """
<div style="margin-bottom: 12px;">
  <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 700;">🧩 Ghép Nhiều File Audio Thành 1 Bản Thu Hoàn Chỉnh</h2>
  <p style="margin: 0; color: #71717a; font-size: 14px;">Tự động nhận diện và sắp xếp thứ tự chính xác theo số tự nhiên trong tên file (vd: <code>seg_1.wav</code> ➔ <code>seg_2.wav</code> ➔ ... ➔ <code>seg_10.wav</code>). Hỗ trợ chèn khoảng lặng giữa các câu để giọng đọc tự nhiên.</p>
</div>
"""
        )
        with gr.Row():
            # Left Card: Input Sources
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 📂 Bước 1: Chọn Nguồn File Âm Thanh")
                    am_mode = gr.Radio(
                        label="Phương thức nạp audio",
                        choices=["Quét thư mục cục bộ (Local Folder)", "Tải lên danh sách file trực tiếp"],
                        value="Quét thư mục cục bộ (Local Folder)",
                    )
                    am_folder = gr.Textbox(
                        label="Đường dẫn thư mục chứa audio",
                        value="audio/no_internet",
                        placeholder="Ví dụ: audio/no_internet hoặc đường dẫn thư mục cache...",
                        visible=True,
                    )
                    am_upload_files = gr.File(
                        label="Kéo thả danh sách file Audio vào đây",
                        file_count="multiple",
                        file_types=["audio"],
                        visible=False,
                    )
                    
                    am_gap = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        value=0.3,
                        step=0.1,
                        label="Khoảng lặng nghỉ giữa các câu (giây)",
                        info="Chèn khoảng im lặng tự nhiên giữa các phân đoạn (khuyên dùng: 0.2s – 0.5s).",
                    )
                    am_merge_btn = gr.Button("🚀 Bắt Đầu Ghép Nối Audio", variant="primary", size="lg")

            # Right Card: Status & Output Audio
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 🔊 Bước 2: Danh Sách Nhận Diện & Kết Quả")
                    am_status = gr.Textbox(
                        label="Báo cáo thứ tự ghép nối",
                        lines=8,
                        placeholder="Danh sách và thứ tự các file sau khi sắp xếp tự nhiên sẽ xuất hiện tại đây...",
                    )
                    am_output_audio = gr.Audio(
                        label="🔊 Audio Đã Ghép Hoàn Chỉnh",
                        type="filepath",
                    )
                    am_download_audio = gr.File(
                        label="💾 Tải File Audio Về Máy (.wav)",
                        visible=True,
                    )

        def _toggle_am_mode(choice):
            if choice == "Quét thư mục cục bộ (Local Folder)":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        am_mode.change(
            _toggle_am_mode,
            inputs=[am_mode],
            outputs=[am_folder, am_upload_files],
        )

        am_merge_btn.click(
            process_audio_merger,
            inputs=[am_mode, am_folder, am_upload_files, am_gap],
            outputs=[am_status, am_output_audio, am_download_audio],
        )

    return {}
