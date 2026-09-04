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
    prev_st = "🔊 Có file nghe thử" if prev_p else "🔇 Không có file nghe thử"
    return f"""**Chi tiết hồ sơ giọng:**
- **Tên hiển thị:** `{disp}` (Tệp lưu: `{name}.pt`)
- **Thời gian tạo:** {saved} | **File nghe thử:** {prev_st}
- **Văn bản tham chiếu:** *"{ref_t}"*
"""


def build_voice_manager_tab(model, _gen):
    """Constructs the Voice Manager Tab UI and internal event listeners."""
    with gr.TabItem("🎙️ Voice Manager / Quản lý Hồ sơ Giọng"):
        gr.Markdown(
            """
### 🎙️ Quản lý & Lưu Trữ Hồ Sơ Giọng Mẫu Cố Định (.pt Embedding)
*Tại đây bạn có thể trích xuất vector giọng nói từ một đoạn âm thanh mẫu (3-10 giây) và lưu lại vĩnh viễn.*
*Các tab **Script Clone**, **Voice Clone** và **Batch Voice Clone** chỉ cần chọn tên giọng để sử dụng ngay mà **không cần upload hay mã hóa lại audio**.*
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                vm_name = gr.Textbox(
                    label="Tên nhân vật / Hồ sơ giọng",
                    placeholder="Ví dụ: Lucan_Energetic, Nam_Ke_Chuyen, Nu_Truyen_Cam...",
                )
                vm_audio = gr.Audio(
                    label="File âm thanh mẫu (3–10 giây)",
                    type="filepath",
                    elem_classes="compact-audio",
                )
                vm_text = gr.Textbox(
                    label="Văn bản giọng mẫu (Tùy chọn)",
                    placeholder="Để trống nếu muốn tự động nhận diện (ASR) hoặc dùng tham chiếu mặc định...",
                )
                vm_save_btn = gr.Button("⚡ Trích xuất & Lưu Hồ Sơ Giọng (.pt)", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### 📂 Danh sách Hồ sơ giọng đã lưu")
                init_profiles = list_voice_profiles()
                init_selected = init_profiles[0] if init_profiles else None
                init_preview = get_voice_profile_preview(init_selected) if init_selected else None

                vm_profiles_list = gr.Dropdown(
                    label="Chọn hồ sơ giọng",
                    choices=init_profiles,
                    value=init_selected,
                    interactive=True,
                )
                with gr.Row():
                    vm_refresh_btn = gr.Button("🔄 Làm mới danh sách", size="sm")
                    vm_delete_btn = gr.Button("🗑️ Xóa hồ sơ này", size="sm", variant="stop")
                
                vm_preview_audio = gr.Audio(
                    label="Nghe thử giọng mẫu gốc đã lưu",
                    value=init_preview,
                    interactive=False,
                    elem_classes="compact-audio"
                )
                vm_info_md = gr.Markdown(value=format_voice_info(init_selected))
                
                with gr.Accordion("🔊 Thử nghiệm đọc văn bản nhanh với giọng này", open=False):
                    with gr.Row():
                        vm_test_text = gr.Textbox(
                            label="Văn bản test giọng",
                            value="Xin chào, đây là giọng đọc thử nghiệm được tạo từ hồ sơ đã lưu.",
                            lines=2,
                            scale=3
                        )
                        vm_test_lang = create_lang_dropdown("Ngôn ngữ", "Auto")
                    vm_test_btn = gr.Button("▶ Sinh giọng đọc thử", variant="secondary")
                    vm_test_audio = gr.Audio(label="Kết quả đọc thử", type="numpy")

                vm_status = gr.Textbox(label="Trạng thái", lines=3)

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
