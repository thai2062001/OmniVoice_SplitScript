import gradio as gr
from omnivoice.webui.config import _ALL_LANGUAGES


def create_lang_dropdown(label="Ngôn ngữ (Language)", value="Auto"):
    """Creates a reusable language selection dropdown."""
    return gr.Dropdown(
        label=label,
        choices=_ALL_LANGUAGES,
        value=value,
        allow_custom_value=False,
        interactive=True,
        info="Mặc định 'Auto' để tự động nhận diện ngôn ngữ.",
    )


def create_gen_settings():
    """Creates a reusable Generation Settings accordion with Turbo Draft toggle."""
    with gr.Accordion("⚙️ Cài đặt tạo giọng nâng cao (Generation Settings)", open=False):
        with gr.Row():
            turbo_draft = gr.Checkbox(
                label="⚡ Chế độ nháp siêu tốc (Turbo Draft - 16 bước)",
                value=False,
                info="Bật để nghe thử nhanh nhịp điệu/cảm xúc kịch bản với tốc độ nhanh nhất (16 steps).",
            )
        with gr.Row():
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Tốc độ nói (Speed)",
                info="1.0 = chuẩn. >1 nói nhanh hơn, <1 nói chậm hơn.",
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
                label="Số bước khử nhiễu (Inference Steps)",
                info="Mặc định: 24 (Tối ưu tốc độ & chất lượng trên GPU Colab/T4).",
                scale=1
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Độ bám sát chỉ dẫn (Guidance Scale / CFG)",
                info="Mặc định: 2.0.",
                scale=1
            )
        with gr.Row():
            dn = gr.Checkbox(
                label="Khử nhiễu nền (Denoise)",
                value=True,
                info="Bật để lọc bớt tạp âm nền.",
            )
            pp = gr.Checkbox(
                label="Tiền xử lý mẫu giọng (Preprocess Prompt)",
                value=True,
                info="Tự động cắt khoảng lặng ở giọng mẫu.",
            )
            po = gr.Checkbox(
                label="Hậu xử lý kết quả (Postprocess Output)",
                value=True,
                info="Xóa bỏ khoảng lặng thừa ở cuối file audio sinh ra.",
            )

        turbo_draft.change(
            lambda is_turbo: 16 if is_turbo else 24,
            inputs=[turbo_draft],
            outputs=[ns]
        )

    return ns, gs, dn, sp, du, pp, po
