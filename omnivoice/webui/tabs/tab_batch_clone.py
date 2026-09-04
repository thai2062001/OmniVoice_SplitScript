import gradio as gr
from omnivoice.webui.profile_manager import list_voice_profiles, get_voice_profile_preview, load_voice_profile
from omnivoice.webui.components import create_lang_dropdown, create_gen_settings


def build_batch_clone_tab(model, _gen):
    """Constructs the Batch Voice Clone Tab UI and internal event listeners."""
    with gr.TabItem("📑 Clone Hàng Loạt (Batch Clone)"):
        gr.Markdown(
            """
<div style="margin-bottom: 12px;">
  <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 700;">📑 Nhân Bản Giọng Nói Hàng Loạt (Batch Clone)</h2>
  <p style="margin: 0; color: #71717a; font-size: 14px;">Tạo giọng đọc cùng lúc cho <b>5 câu văn bản riêng biệt</b> bằng chung 1 mẫu giọng đã chọn.</p>
</div>
"""
        )
        with gr.Row():
            # Left Card: Voice Source & Common Settings
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 🎙️ Bước 1: Chọn Giọng Mẫu Dùng Chung")
                    bvc_source_type = gr.Radio(
                        choices=["🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)", "📤 Tải lên Audio mẫu mới"],
                        value="🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if list_voice_profiles() else "📤 Tải lên Audio mẫu mới",
                        label="Nguồn giọng đọc"
                    )

                    with gr.Group(visible=bool(list_voice_profiles())) as bvc_preset_group:
                        with gr.Row():
                            bvc_saved_profile = gr.Dropdown(
                                label="Chọn hồ sơ giọng",
                                choices=list_voice_profiles(),
                                value=list_voice_profiles()[0] if list_voice_profiles() else None,
                                scale=3
                            )
                            bvc_preset_preview = gr.Audio(
                                label="Nghe thử",
                                value=get_voice_profile_preview(list_voice_profiles()[0]) if list_voice_profiles() else None,
                                interactive=False,
                                scale=2,
                                elem_classes="compact-audio"
                            )

                    with gr.Group(visible=not bool(list_voice_profiles())) as bvc_custom_group:
                        bvc_ref_audio = gr.Audio(
                            label="Tải lên file âm thanh giọng mẫu (3–10 giây)",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        bvc_ref_text = gr.Textbox(
                            label="Văn bản giọng mẫu (Tùy chọn)",
                            placeholder="Để trống nếu muốn tự động nhận diện (ASR)...",
                        )

                    def _on_bvc_source_change(mode_choice):
                        is_preset = "Hồ sơ giọng có sẵn" in mode_choice
                        return gr.update(visible=is_preset), gr.update(visible=not is_preset)

                    bvc_source_type.change(
                        _on_bvc_source_change,
                        inputs=[bvc_source_type],
                        outputs=[bvc_preset_group, bvc_custom_group]
                    )
                    bvc_saved_profile.change(
                        lambda p: get_voice_profile_preview(p),
                        inputs=[bvc_saved_profile],
                        outputs=[bvc_preset_preview]
                    )

                    bvc_lang = create_lang_dropdown("Ngôn ngữ giọng đọc (Language)")

                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### ⚙️ Bước 2: Cài Đặt Nâng Cao Dùng Chung")
                    with gr.Accordion("Chỉ dẫn phong cách biểu cảm (Tùy chọn)", open=False):
                        bvc_instruct = gr.Textbox(label="Chỉ dẫn biểu cảm (Instruct)", placeholder="Ví dụ: whisper, high pitch, low pitch...", lines=2)
                    (
                        bvc_ns,
                        bvc_gs,
                        bvc_dn,
                        bvc_sp,
                        bvc_du,
                        bvc_pp,
                        bvc_po,
                    ) = create_gen_settings()

            # Right Card: 5 Text Inputs & 5 Outputs
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 📝 Bước 3: Nhập 5 Đoạn Văn Bản & Kết Quả")
                    
                    with gr.Group(elem_classes="segment-card"):
                        bvc_text1 = gr.Textbox(label="Câu 1", placeholder="Nhập câu số 1...", lines=2)
                        bvc_audio1 = gr.Audio(label="🔊 Audio Câu 1", type="numpy")
                    
                    with gr.Group(elem_classes="segment-card"):
                        bvc_text2 = gr.Textbox(label="Câu 2", placeholder="Nhập câu số 2...", lines=2)
                        bvc_audio2 = gr.Audio(label="🔊 Audio Câu 2", type="numpy")
                    
                    with gr.Group(elem_classes="segment-card"):
                        bvc_text3 = gr.Textbox(label="Câu 3", placeholder="Nhập câu số 3...", lines=2)
                        bvc_audio3 = gr.Audio(label="🔊 Audio Câu 3", type="numpy")
                    
                    with gr.Group(elem_classes="segment-card"):
                        bvc_text4 = gr.Textbox(label="Câu 4", placeholder="Nhập câu số 4...", lines=2)
                        bvc_audio4 = gr.Audio(label="🔊 Audio Câu 4", type="numpy")
                    
                    with gr.Group(elem_classes="segment-card"):
                        bvc_text5 = gr.Textbox(label="Câu 5", placeholder="Nhập câu số 5...", lines=2)
                        bvc_audio5 = gr.Audio(label="🔊 Audio Câu 5", type="numpy")
                    
                    bvc_btn = gr.Button("🚀 Bắt Đầu Tạo Giọng Hàng Loạt (5 Câu)", variant="primary", size="lg")
                    bvc_status = gr.Textbox(label="Trạng thái & Tiến trình", lines=4)

        def _batch_clone_fn(
            lang, 
            source_type,
            saved_prof,
            ref_audio,
            ref_text,
            text1,
            text2,
            text3,
            text4,
            text5,
            instruct, ns, gs, dn, sp, du, pp, po
        ):
            results = []
            statuses = []
            
            prompt = None
            actual_ref_audio = ref_audio
            if "Hồ sơ giọng có sẵn" in source_type:
                if not saved_prof:
                    return None, None, None, None, None, "❌ Lỗi: Vui lòng chọn một hồ sơ giọng đã lưu từ danh sách."
                prompt, _ = load_voice_profile(saved_prof)
                actual_ref_audio = None
                if prompt is None:
                    return None, None, None, None, None, f"❌ Lỗi: Không tìm thấy hồ sơ {saved_prof}.pt"
            elif ref_audio and str(ref_audio).strip():
                try:
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=ref_audio,
                        ref_text=ref_text or None,
                    )
                except Exception as e:
                    return None, None, None, None, None, f"❌ Lỗi trích xuất audio mẫu: {e}"
            else:
                return None, None, None, None, None, "❌ Lỗi: Vui lòng chọn hồ sơ giọng có sẵn hoặc tải lên file âm thanh mẫu."
            
            texts = [text1, text2, text3, text4, text5]
            
            for i, t in enumerate(texts, 1):
                if t and t.strip():
                    try:
                        res, stat = _gen(
                            t.strip(),
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
                            voice_clone_prompt=prompt,
                        )
                        results.append(res)
                        statuses.append(f"Voice {i}: {stat}")
                    except Exception as e:
                        results.append(None)
                        statuses.append(f"Voice {i}: Error: {e}")
                else:
                    results.append(None)
                    statuses.append(f"Voice {i}: Skipped (empty text)")
                    
            return results[0], results[1], results[2], results[3], results[4], "\n".join(statuses)

        bvc_btn.click(
            _batch_clone_fn,
            inputs=[
                bvc_lang,
                bvc_source_type,
                bvc_saved_profile,
                bvc_ref_audio,
                bvc_ref_text,
                bvc_text1,
                bvc_text2,
                bvc_text3,
                bvc_text4,
                bvc_text5,
                bvc_instruct,
                bvc_ns,
                bvc_gs,
                bvc_dn,
                bvc_sp,
                bvc_du,
                bvc_pp,
                bvc_po,
            ],
            outputs=[bvc_audio1, bvc_audio2, bvc_audio3, bvc_audio4, bvc_audio5, bvc_status],
        )

    return {
        "bvc_source_type": bvc_source_type,
        "bvc_preset_group": bvc_preset_group,
        "bvc_custom_group": bvc_custom_group,
        "bvc_saved_profile": bvc_saved_profile,
        "bvc_preset_preview": bvc_preset_preview,
    }
