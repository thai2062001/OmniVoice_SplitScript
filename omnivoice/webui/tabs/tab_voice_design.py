import gradio as gr
from omnivoice.webui.config import _CATEGORIES, _ATTR_INFO
from omnivoice.webui.components import create_lang_dropdown, create_gen_settings


def build_voice_design_tab(_gen):
    """Constructs the Voice Design Tab UI and internal event listeners."""
    with gr.TabItem("🎨 Thiết Kế Giọng (Voice Design)"):
        gr.Markdown(
            """
### 🎨 Thiết Kế Giọng Nói AI Hoàn Toàn Mới (Voice Design)
*Tạo giọng đọc nhân tạo từ số 0 bằng cách kết hợp các đặc tính (Giới tính, Độ tuổi, Âm điệu, Phong cách, Accent) mà **không cần audio mẫu**.*
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                vd_text = gr.Textbox(
                    label="Văn bản cần đọc (Text to Synthesize)",
                    lines=4,
                    placeholder="Nhập nội dung bạn muốn giọng AI đọc vào đây...",
                )
                vd_lang = create_lang_dropdown("Ngôn ngữ (Language)")

                _AUTO = "Tự động (Auto)"
                vd_groups = []
                for _cat, _choices in _CATEGORIES.items():
                    vd_groups.append(
                        gr.Dropdown(
                            label=_cat,
                            choices=[_AUTO] + _choices,
                            value=_AUTO,
                            info=_ATTR_INFO.get(_cat),
                        )
                    )

                (
                    vd_ns,
                    vd_gs,
                    vd_dn,
                    vd_sp,
                    vd_du,
                    vd_pp,
                    vd_po,
                ) = create_gen_settings()
                vd_btn = gr.Button("🚀 Bắt Đầu Tạo Giọng (Voice Design)", variant="primary")
            with gr.Column(scale=1):
                vd_audio = gr.Audio(
                    label="🔊 Kết Quả Âm Thanh",
                    type="numpy",
                )
                vd_status = gr.Textbox(label="Trạng thái", lines=2)

        def _build_instruct(groups):
            """Extract instruct text from UI dropdowns."""
            selected = [g for g in groups if g and g != "Tự động (Auto)" and g != "Auto"]
            if not selected:
                return None
            parts = []
            for v in selected:
                # Extract English tag in parentheses if available e.g. "Nam (Male)" -> "Male"
                import re
                match_paren = re.search(r'\(([^)]+)\)', v)
                if match_paren:
                    inner = match_paren.group(1).strip()
                    if " / " in inner:
                        # Chinese dialect: "Henan Dialect / 河南话"
                        parts.append(inner.split(" / ")[-1].strip())
                    else:
                        parts.append(inner)
                elif " / " in v:
                    parts.append(v.split(" / ")[0].strip())
                else:
                    parts.append(v)
            return ", ".join(parts)

        def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
            if not text or not text.strip():
                return None, "Vui lòng nhập văn bản cần tổng hợp giọng nói."
            return _gen(
                text.strip(),
                lang,
                None,
                _build_instruct(groups),
                ns,
                gs,
                dn,
                sp,
                du,
                pp,
                po,
                mode="design",
            )

        vd_btn.click(
            _design_fn,
            inputs=[
                vd_text,
                vd_lang,
                vd_ns,
                vd_gs,
                vd_dn,
                vd_sp,
                vd_du,
                vd_pp,
                vd_po,
            ]
            + vd_groups,
            outputs=[vd_audio, vd_status],
        )

    return {}
