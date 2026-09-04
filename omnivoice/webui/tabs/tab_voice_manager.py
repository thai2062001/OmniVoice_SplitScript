import os
import gradio as gr
from omnivoice.webui.profile_manager import (
    list_voice_profiles,
    get_voice_profile_metadata,
    get_voice_profile_preview,
    load_voice_profile,
)
from omnivoice.webui.components import create_lang_dropdown


def format_voice_info(name):
    """Formats markdown information for a selected voice profile."""
    if not name:
        return "*(Chưa có hồ sơ giọng nào được chọn)*"
    meta = get_voice_profile_metadata(name)
    disp = meta.get("display_name", name)
    saved = meta.get("saved_at", "Không rõ")
    ref_t = meta.get("ref_text", "*(Không có văn bản tham chiếu)*")
    prev_p = get_voice_profile_preview(name)
    prev_st = "🟢 Có sẵn audio mẫu" if prev_p else "⚪ Không có audio mẫu"
    return f"""
<div style="background: rgba(0,0,0,0.02); border: 1px solid #e4e4e7; border-radius: 10px; padding: 14px 16px; margin: 10px 0;">
  <div style="font-weight: 700; font-size: 15px; margin-bottom: 6px;">👤 {disp} <span style="font-size: 12px; font-weight: normal; color: #71717a;">({name}.pt)</span></div>
  <div style="font-size: 13px; color: #52525b; margin-bottom: 4px;">📅 <b>Ngày tạo:</b> {saved} &nbsp;|&nbsp; <b>Trạng thái:</b> {prev_st}</div>
  <div style="font-size: 13px; color: #52525b;">📝 <b>Nội dung mẫu:</b> <i>"{ref_t}"</i></div>
</div>
"""


def build_voice_manager_tab(model, _gen):
    """Constructs the Voice Manager Tab UI and internal event listeners."""
    with gr.TabItem("🎙️ Quản Lý Hồ Sơ Giọng (Voice Manager)"):
        gr.Markdown(
            """
<div style="margin-bottom: 12px;">
  <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 700;">🎙️ Quản Lý & Lưu Trữ Hồ Sơ Giọng Mẫu (.pt)</h2>
  <p style="margin: 0; color: #71717a; font-size: 14px;">Trích xuất đặc trưng giọng nói từ đoạn ghi âm <b>3–10 giây</b> và lưu cố định. Giúp bạn sử dụng lại ngay lập tức ở tất cả các tab khác mà <b>không cần upload lại audio</b>.</p>
</div>
"""
        )
        with gr.Row():
            # Left Card: Create New Profile
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### ➕ Bước 1: Tạo Hồ Sơ Giọng Mới")
                    vm_name = gr.Textbox(
                        label="Tên nhận diện nhân vật / giọng đọc",
                        placeholder="Ví dụ: Nam_Ke_Chuyen, Nu_Truyen_Cam, MC_Sot_Sang...",
                    )
                    vm_audio = gr.Audio(
                        label="Tải lên file âm thanh giọng mẫu (3–10 giây)",
                        type="filepath",
                        elem_classes="compact-audio",
                    )
                    vm_text = gr.Textbox(
                        label="Văn bản giọng mẫu (Tùy chọn - Tăng độ chính xác)",
                        placeholder="Nhập câu người trong audio nói. Để trống nếu muốn Whisper AI tự nhận diện...",
                        lines=2,
                    )
                    vm_save_btn = gr.Button("⚡ Trích Xuất & Lưu Hồ Sơ Giọng (.pt)", variant="primary")

            # Right Card: Library & Quick Test
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 📂 Bước 2: Thư Viện Giọng Đã Lưu & Nghe Thử")
                    init_profiles = list_voice_profiles()
                    init_selected = init_profiles[0] if init_profiles else None
                    init_preview = get_voice_profile_preview(init_selected) if init_selected else None

                    with gr.Row():
                        vm_profiles_list = gr.Dropdown(
                            label="Danh sách hồ sơ giọng hiện có",
                            choices=init_profiles,
                            value=init_selected,
                            interactive=True,
                            scale=3
                        )
                        vm_refresh_btn = gr.Button("🔄 Làm mới", size="sm", scale=1)
                        vm_delete_btn = gr.Button("🗑️ Xóa", size="sm", variant="stop", scale=1)
                    
                    vm_info_md = gr.Markdown(value=format_voice_info(init_selected))

                    vm_preview_audio = gr.Audio(
                        label="🔊 Audio mẫu gốc của hồ sơ đang chọn",
                        value=init_preview,
                        interactive=False,
                        elem_classes="compact-audio"
                    )
                    
                    with gr.Accordion("▶ Đọc thử nghiệm câu bất kỳ với giọng này", open=True):
                        with gr.Row():
                            vm_test_text = gr.Textbox(
                                label="Văn bản test",
                                value="Xin chào các bạn, đây là giọng đọc thử nghiệm từ hồ sơ giọng đã lưu.",
                                lines=2,
                                scale=3
                            )
                            vm_test_lang = create_lang_dropdown("Ngôn ngữ", "Auto")
                        vm_test_btn = gr.Button("▶ Tạo Giọng Đọc Thử", variant="secondary")
                        vm_test_audio = gr.Audio(label="🔊 Kết quả đọc thử", type="numpy")

                    vm_status = gr.Textbox(label="Trạng thái hệ thống", lines=2)

        def _on_vm_select(profile_nm):
            if not profile_nm:
                return None, "*(Chưa có hồ sơ giọng nào được chọn)*", ""
            preview_aud = get_voice_profile_preview(profile_nm)
            info_text = format_voice_info(profile_nm)
            return preview_aud, info_text, f"Đã chọn hồ sơ giọng: {profile_nm}"

        def _on_vm_quick_test(profile_nm, test_text, lang):
            if not profile_nm:
                return None, "Vui lòng chọn hồ sơ giọng trước khi đọc thử."
            if not test_text or not test_text.strip():
                return None, "Vui lòng nhập văn bản đọc thử."
            loaded_prompt, _ = load_voice_profile(profile_nm)
            if loaded_prompt is None:
                return None, f"Lỗi: Không tìm thấy file hồ sơ {profile_nm}.pt"
            return _gen(
                test_text.strip(),
                lang,
                None,
                None,
                24,
                2.0,
                True,
                1.0,
                None,
                True,
                True,
                mode="clone",
                ref_text=None,
                voice_clone_prompt=loaded_prompt
            )

        vm_profiles_list.change(
            _on_vm_select,
            inputs=[vm_profiles_list],
            outputs=[vm_preview_audio, vm_info_md, vm_status],
        )
        vm_test_btn.click(
            _on_vm_quick_test,
            inputs=[vm_profiles_list, vm_test_text, vm_test_lang],
            outputs=[vm_test_audio, vm_status]
        )

    return {
        "vm_name": vm_name,
        "vm_audio": vm_audio,
        "vm_text": vm_text,
        "vm_save_btn": vm_save_btn,
        "vm_profiles_list": vm_profiles_list,
        "vm_refresh_btn": vm_refresh_btn,
        "vm_delete_btn": vm_delete_btn,
        "vm_preview_audio": vm_preview_audio,
        "vm_info_md": vm_info_md,
        "vm_status": vm_status,
    }
