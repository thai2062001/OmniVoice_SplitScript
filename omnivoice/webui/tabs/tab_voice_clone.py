import gradio as gr
from omnivoice.webui.profile_manager import list_voice_profiles, get_voice_profile_preview, load_voice_profile
from omnivoice.webui.components import create_lang_dropdown, create_gen_settings


def build_voice_clone_tab(_gen):
    """Constructs the Voice Clone Tab UI and internal event listeners."""
    with gr.TabItem("👤 Nhân Bản Giọng (Voice Clone)"):
        gr.Markdown(
            """
### 👤 Nhân Bản Giọng Nói Đơn Lẻ (Voice Clone)
*Tạo giọng đọc mô phỏng theo mẫu giọng đã lưu hoặc tải lên đoạn âm thanh ngắn (3-10s).*
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                vc_text = gr.Textbox(
                    label="Văn bản cần đọc (Text to Synthesize)",
                    lines=4,
                    placeholder="Nhập đoạn văn bản bạn muốn chuyển thành giọng nói...",
                )

                vc_source_type = gr.Radio(
                    choices=["🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)", "📤 Tải lên Audio mẫu mới"],
                    value="🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if list_voice_profiles() else "📤 Tải lên Audio mẫu mới",
                    label="Nguồn Giọng Mẫu (Voice Source)"
                )

                with gr.Group(visible=bool(list_voice_profiles())) as vc_preset_group:
                    with gr.Row():
                        vc_saved_profile = gr.Dropdown(
                            label="Chọn Hồ sơ giọng cố định",
                            choices=list_voice_profiles(),
                            value=list_voice_profiles()[0] if list_voice_profiles() else None,
                            scale=3
                        )
                        vc_preset_preview = gr.Audio(
                            label="Nghe thử",
                            value=get_voice_profile_preview(list_voice_profiles()[0]) if list_voice_profiles() else None,
                            interactive=False,
                            scale=2,
                            elem_classes="compact-audio"
                        )

                with gr.Group(visible=not bool(list_voice_profiles())) as vc_custom_group:
                    vc_ref_audio = gr.Audio(
                        label="File âm thanh giọng mẫu (Reference Audio)",
                        type="filepath",
                        elem_classes="compact-audio",
                    )
                    gr.Markdown(
                        "<span style='font-size:0.85em;color:#888;'>"
                        "Khuyến nghị: File audio dài 3–10 giây, giọng đọc rõ ràng."
                        "</span>"
                    )
                    vc_ref_text = gr.Textbox(
                        label="Văn bản của giọng mẫu (Tùy chọn)",
                        lines=2,
                        placeholder="Nội dung người trong audio nói. Để trống nếu muốn tự động nhận diện (ASR).",
                    )

                def _on_vc_source_change(mode_choice):
                    is_preset = "Hồ sơ giọng có sẵn" in mode_choice
                    return gr.update(visible=is_preset), gr.update(visible=not is_preset)

                vc_source_type.change(
                    _on_vc_source_change,
                    inputs=[vc_source_type],
                    outputs=[vc_preset_group, vc_custom_group]
                )
                vc_saved_profile.change(
                    lambda p: get_voice_profile_preview(p),
                    inputs=[vc_saved_profile],
                    outputs=[vc_preset_preview]
                )

                vc_lang = create_lang_dropdown("Ngôn ngữ (Language)")
                with gr.Accordion("Chỉ dẫn phong cách / Cảm xúc (Tùy chọn)", open=False):
                    vc_instruct = gr.Textbox(label="Chỉ dẫn phong cách (Instruct)", placeholder="Ví dụ: whisper, high pitch, low pitch, dramatic...", lines=2)
                (
                    vc_ns,
                    vc_gs,
                    vc_dn,
                    vc_sp,
                    vc_du,
                    vc_pp,
                    vc_po,
                ) = create_gen_settings()
                vc_btn = gr.Button("🚀 Bắt Đầu Tạo Giọng (Voice Clone)", variant="primary")
            with gr.Column(scale=1):
                vc_audio = gr.Audio(
                    label="🔊 Kết Quả Âm Thanh",
                    type="numpy",
                )
                vc_status = gr.Textbox(label="Trạng thái", lines=2)

        def _clone_fn(
            text, lang, source_type, saved_prof, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
        ):
            loaded_prompt = None
            actual_ref_audio = ref_aud
            if "Hồ sơ giọng có sẵn" in source_type:
                if not saved_prof:
                    return None, "❌ Lỗi: Vui lòng chọn một hồ sơ giọng đã lưu từ danh sách."
                loaded_prompt, _ = load_voice_profile(saved_prof)
                actual_ref_audio = None
                if loaded_prompt is None:
                    return None, f"❌ Lỗi: Không tìm thấy file hồ sơ {saved_prof}.pt"

            return _gen(
                text,
                lang,
                actual_ref_audio,
                instruct,
                ns,
                gs,
                dn,
                sp,
                du,
                pp,
                po,
                mode="clone",
                ref_text=ref_text or None,
                voice_clone_prompt=loaded_prompt,
            )

        vc_btn.click(
            _clone_fn,
            inputs=[
                vc_text,
                vc_lang,
                vc_source_type,
                vc_saved_profile,
                vc_ref_audio,
                vc_ref_text,
                vc_instruct,
                vc_ns,
                vc_gs,
                vc_dn,
                vc_sp,
                vc_du,
                vc_pp,
                vc_po,
            ],
            outputs=[vc_audio, vc_status],
        )

    return {
        "vc_source_type": vc_source_type,
        "vc_preset_group": vc_preset_group,
        "vc_custom_group": vc_custom_group,
        "vc_saved_profile": vc_saved_profile,
        "vc_preset_preview": vc_preset_preview,
    }
