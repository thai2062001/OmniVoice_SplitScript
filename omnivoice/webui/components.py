import gradio as gr
from omnivoice.webui.config import _ALL_LANGUAGES


def create_lang_dropdown(label="Ngôn ngữ đọc", value="Tự động (Auto)"):
    """Creates a reusable language selection dropdown."""
    return gr.Dropdown(
        label=label,
        choices=_ALL_LANGUAGES,
        value=value,
        allow_custom_value=False,
        interactive=True,
        info="Chọn 'Tự động' để hệ thống tự nhận diện ngôn ngữ của câu.",
    )


def create_gen_settings():
    """Creates a reusable Generation Settings accordion with Turbo Draft toggle."""
    with gr.Accordion("⚙️ Cài đặt tạo giọng nâng cao", open=False):
        with gr.Row():
            turbo_draft = gr.Checkbox(
                label="⚡ Chế độ nháp siêu tốc (16 bước)",
                value=False,
                info="Bật để nghe thử nhanh nhịp điệu và cảm xúc kịch bản với tốc độ nhanh nhất (16 bước).",
            )
        with gr.Row():
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Tốc độ đọc",
                info="1.0 = chuẩn. >1 đọc nhanh hơn, <1 đọc chậm hơn.",
                scale=1
            )
            du = gr.Number(
                value=None,
                label="Thời lượng cố định (giây)",
                info="Để trống để dùng tốc độ. Điền số để ép đúng số giây mong muốn.",
                scale=1
            )
        with gr.Row():
            ns = gr.Slider(
                4,
                64,
                value=24,
                step=1,
                label="Số bước khử nhiễu",
                info="Mặc định: 24 (Tối ưu độ tự nhiên & tốc độ trên GPU Colab/T4).",
                scale=1
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Độ bám sát chỉ dẫn biểu cảm",
                info="Mặc định: 2.0 (Mức độ tuân thủ cảm xúc và phong cách).",
                scale=1
            )
        with gr.Row():
            dn = gr.Checkbox(
                label="Khử nhiễu âm thanh nền",
                value=True,
                info="Bật để lọc bớt tạp âm nền và làm trong giọng.",
            )
            pp = gr.Checkbox(
                label="Tiền xử lý mẫu giọng",
                value=True,
                info="Tự động cắt tỉa khoảng lặng thừa ở audio mẫu.",
            )
            po = gr.Checkbox(
                label="Hậu xử lý âm thanh đầu ra",
                value=True,
                info="Tự động làm sạch và xóa khoảng lặng thừa ở cuối câu.",
            )

        turbo_draft.change(
            lambda is_turbo: 16 if is_turbo else 24,
            inputs=[turbo_draft],
            outputs=[ns]
        )

    return ns, gs, dn, sp, du, pp, po
