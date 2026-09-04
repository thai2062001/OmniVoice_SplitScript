import gradio as gr
from omnivoice.webui.config import _CATEGORIES, _ATTR_INFO
from omnivoice.webui.components import create_lang_dropdown, create_gen_settings


def build_voice_design_tab(_gen):
    """Constructs the Voice Design Tab UI and internal event listeners."""
    with gr.TabItem("🎨 Thiết Kế Giọng (Voice Design)"):
        gr.Markdown(
            """
<div style="margin-bottom: 12px;">
  <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 700;">🎨 Thiết Kế Giọng Nói AI Mới (Voice Design)</h2>
  <p style="margin: 0; color: #71717a; font-size: 14px;">Tạo một giọng đọc hoàn toàn mới từ số 0 bằng cách phối hợp các đặc tính (Giới tính, Độ tuổi, Âm điệu, Phong cách, Accent) mà <b>không cần bất kỳ audio mẫu nào</b>.</p>
</div>
"""
        )
        with gr.Row():
            # Left Card: Attributes & Text
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 📝 Bước 1: Nội Dung Cần Đọc")
                    vd_text = gr.Textbox(
                        label="Văn bản",
                        lines=4,
                        placeholder="Nhập nội dung bạn muốn giọng AI thiết kế đọc vào đây...",
                    )
                    vd_lang = create_lang_dropdown("Ngôn ngữ (Language)")

                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 🎭 Bước 2: Thiết Lập Đặc Tính Giọng Nói")
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

                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### ⚙️ Bước 3: Cài Đặt Sinh Giọng")
                    (
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ) = create_gen_settings()
                    vd_btn = gr.Button("🚀 Bắt Đầu Thiết Kế & Tạo Giọng", variant="primary", size="lg")

            # Right Card: Output
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 🔊 Kết Quả Âm Thanh")
                    vd_audio = gr.Audio(
                        label="File âm thanh đã thiết kế",
                        type="numpy",
                    )
                    vd_status = gr.Textbox(label="Trạng thái & Tiến trình", lines=3)

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
