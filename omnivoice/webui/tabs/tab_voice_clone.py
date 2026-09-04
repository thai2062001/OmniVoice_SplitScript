import gradio as gr
from omnivoice.webui.profile_manager import list_voice_profiles, get_voice_profile_preview, load_voice_profile
from omnivoice.webui.components import create_lang_dropdown, create_gen_settings


def build_voice_clone_tab(_gen):
    """Constructs the Voice Clone Tab UI and internal event listeners."""
    with gr.TabItem("👤 Nhân Bản Giọng (Voice Clone)"):
        gr.Markdown(
            """
<div style="margin-bottom: 12px;">
  <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 700;">👤 Nhân Bản Giọng Nói Đơn Lẻ (Voice Clone)</h2>
  <p style="margin: 0; color: #71717a; font-size: 14px;">Mô phỏng chính xác chất giọng từ hồ sơ đã lưu hoặc file ghi âm ngắn (3-10s) và đọc bất kỳ văn bản nào.</p>
</div>
"""
        )
        with gr.Row():
            # Left Card: Input & Source Selection
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 📝 Bước 1: Văn Bản Cần Đọc")
                    vc_text = gr.Textbox(
                        label="Văn bản",
                        lines=4,
                        placeholder="Nhập nội dung bạn muốn giọng AI đọc vào đây...",
                    )
                    with gr.Row():
                        vc_copy_btn = gr.Button("📋 Sao chép văn bản", size="sm", elem_classes="btn-copy-action", scale=1)
                        vc_char_count = gr.Markdown("📊 Số ký tự: **0** | Ước tính độ dài: **0s**", elem_classes="char-counter")
                    vc_lang = create_lang_dropdown("Ngôn ngữ giọng đọc (Language)")

                    def _update_vc_char_counter(txt):
                        txt = txt or ""
                        char_len = len(txt)
                        word_len = len(txt.split()) if txt else 0
                        # Khoảng 3 từ / giây
                        est_sec = round(word_len / 2.8, 1) if word_len > 0 else 0
                        return f"📊 **{char_len}** ký tự ({word_len} từ) | Ước tính: **~{est_sec}s**"

                    vc_text.change(
                        _update_vc_char_counter,
                        inputs=[vc_text],
                        outputs=[vc_char_count]
                    )
                    vc_copy_btn.click(
                        None,
                        inputs=[vc_text],
                        outputs=None,
                        js="(txt) => { window.copyTextToClipboard(txt, '📋 Đã sao chép văn bản vào bộ nhớ tạm!'); }"
                    )

                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 🎙️ Bước 2: Nguồn Giọng Mẫu")
                    vc_source_type = gr.Radio(
                        choices=["🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)", "📤 Tải lên Audio mẫu mới"],
                        value="🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if list_voice_profiles() else "📤 Tải lên Audio mẫu mới",
                        label="Lựa chọn nguồn giọng"
                    )

                    with gr.Group(visible=bool(list_voice_profiles())) as vc_preset_group:
                        with gr.Row():
                            vc_saved_profile = gr.Dropdown(
                                label="Chọn hồ sơ giọng",
                                choices=list_voice_profiles(),
                                value=list_voice_profiles()[0] if list_voice_profiles() else None,
                                scale=3
                            )
                            vc_preset_preview = gr.Audio(
                                label="Nghe thử giọng mẫu",
                                value=get_voice_profile_preview(list_voice_profiles()[0]) if list_voice_profiles() else None,
                                interactive=False,
                                scale=2,
                                elem_classes="compact-audio"
                            )

                    with gr.Group(visible=not bool(list_voice_profiles())) as vc_custom_group:
                        vc_ref_audio = gr.Audio(
                            label="Tải lên file âm thanh giọng mẫu (3–10 giây)",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        vc_ref_text = gr.Textbox(
                            label="Văn bản giọng mẫu (Tùy chọn - Tăng độ chính xác)",
                            lines=2,
                            placeholder="Nội dung người trong audio nói. Để trống nếu muốn Whisper AI tự nhận diện...",
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

                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### ⚙️ Bước 3: Tùy Chỉnh Nâng Cao")
                    with gr.Accordion("Chỉ dẫn phong cách / Cảm xúc (Tùy chọn)", open=False):
                        vc_instruct = gr.Textbox(
                            label="Chỉ dẫn biểu cảm (Instruct)",
                            placeholder="Ví dụ: whisper (thì thầm), high pitch (sôi nổi), low pitch (trầm ấm)...",
                            lines=2
                        )
                    (
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ) = create_gen_settings()
                    vc_btn = gr.Button("🚀 Bắt Đầu Tạo Giọng Nói", variant="primary", size="lg")

            # Right Card: Output & Player
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 🔊 Kết Quả Âm Thanh")
                    vc_audio = gr.Audio(
                        label="File âm thanh đã tạo",
                        type="numpy",
                    )
                    vc_status = gr.Textbox(
                        label="Trạng thái & Tiến trình",
                        lines=3,
                        value="🟢 Hệ thống sẵn sàng tạo âm thanh."
                    )

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
