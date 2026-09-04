import gradio as gr
from omnivoice.webui.config import _CATEGORIES, _ATTR_INFO
from omnivoice.webui.components import create_lang_dropdown, create_gen_settings


def build_voice_design_tab(_gen):
    """Constructs the Voice Design Tab UI and internal event listeners."""
    with gr.TabItem("Voice Design"):
        with gr.Row():
            with gr.Column(scale=1):
                vd_text = gr.Textbox(
                    label="Text to Synthesize / 待合成文本",
                    lines=4,
                    placeholder="Enter the text you want to synthesize...",
                )
                vd_lang = create_lang_dropdown()

                _AUTO = "Auto"
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
                vd_btn = gr.Button("Generate / 生成", variant="primary")
            with gr.Column(scale=1):
                vd_audio = gr.Audio(
                    label="Output Audio / 合成结果",
                    type="numpy",
                )
                vd_status = gr.Textbox(label="Status / 状态", lines=2)

        def _build_instruct(groups):
            """Extract instruct text from UI dropdowns."""
            selected = [g for g in groups if g and g != "Auto"]
            if not selected:
                return None
            parts = []
            for v in selected:
                if " / " in v:
                    en, zh = v.split(" / ", 1)
                    # Dialects have no English equivalent
                    if "Dialect" in v.split(" / ")[0]:
                        parts.append(zh.strip())
                    else:
                        parts.append(en.strip())
                else:
                    parts.append(v)
            return ", ".join(parts)

        def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
            return _gen(
                text,
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
