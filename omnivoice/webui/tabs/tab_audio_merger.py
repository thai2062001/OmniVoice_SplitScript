import gradio as gr
from omnivoice.webui.audio_engine import process_audio_merger


def build_audio_merger_tab():
    """Constructs the Audio Merger Tab UI and internal event listeners."""
    with gr.TabItem("🧩 Ghép Audio"):
        gr.Markdown(
            """
### 🧩 Ghép Nhiều File Audio Thành 1 File Âm Thanh Hoàn Chỉnh
*Tự động phát hiện và sắp xếp thứ tự chính xác theo số trong tên file (ví dụ: `audio_1.wav` ➔ `audio_2.wav` ➔ ... ➔ `audio_10.wav`).*
*Hỗ trợ tải lên danh sách file hoặc quét trực tiếp thư mục (mặc định: `audio/no_internet`).*
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                am_mode = gr.Radio(
                    label="Nguồn file Audio",
                    choices=["Quét thư mục cục bộ (Local Folder)", "Upload file trực tiếp"],
                    value="Quét thư mục cục bộ (Local Folder)",
                )
                am_folder = gr.Textbox(
                    label="Đường dẫn thư mục chứa audio",
                    value="audio/no_internet",
                    placeholder="Ví dụ: audio/no_internet hoặc C:/path/to/audios",
                    visible=True,
                )
                am_upload_files = gr.File(
                    label="Tải lên danh sách file Audio",
                    file_count="multiple",
                    file_types=["audio"],
                    visible=False,
                )
                
                am_gap = gr.Slider(
                    minimum=0.0,
                    maximum=3.0,
                    value=0.3,
                    step=0.1,
                    label="Khoảng lặng giữa các phân đoạn (giây)",
                    info="Chèn thêm khoảng im lặng ngắn giữa các audio để giọng đọc tự nhiên hơn.",
                )
                am_merge_btn = gr.Button("🚀 Bắt Đầu Ghép Audio", variant="primary")

            with gr.Column(scale=1):
                am_status = gr.Textbox(
                    label="Trạng thái & Thứ tự các file đã nhận diện",
                    lines=8,
                    placeholder="Thông tin thứ tự các file sau khi sắp xếp sẽ hiển thị tại đây...",
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
