import os
import logging
import argparse
import subprocess
import threading
import re
import urllib.request
import gradio as gr
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.webui.config import get_theme_and_css, get_head_js, _IS_GDRIVE
from omnivoice.webui.profile_manager import (
    list_voice_profiles,
    get_voice_profile_preview,
    save_voice_profile,
    delete_voice_profile,
)
from omnivoice.webui.audio_engine import extract_voice_prompt_safely
from omnivoice.webui.tabs.tab_voice_manager import build_voice_manager_tab, format_voice_info
from omnivoice.webui.tabs.tab_voice_clone import build_voice_clone_tab
from omnivoice.webui.tabs.tab_batch_clone import build_batch_clone_tab
from omnivoice.webui.tabs.tab_script_clone import build_script_clone_tab
from omnivoice.webui.tabs.tab_voice_design import build_voice_design_tab
from omnivoice.webui.tabs.tab_audio_merger import build_audio_merger_tab
from omnivoice.webui.tabs.tab_guide import build_guide_tab


def start_cloudflare_tunnel(port: int):
    """Starts a Cloudflare Quick Tunnel in the background and prints the high-speed public URL."""
    import platform
    import shutil

    cf_bin = shutil.which("cloudflared")
    if not cf_bin:
        # Download cloudflared if running on Linux (Colab)
        if platform.system().lower() == "linux":
            try:
                logging.info("Downloading cloudflared binary for Linux/Colab...")
                urllib.request.urlretrieve(
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                    "/tmp/cloudflared"
                )
                os.chmod("/tmp/cloudflared", 0o777)
                cf_bin = "/tmp/cloudflared"
            except Exception as e:
                logging.warning(f"Could not auto-download cloudflared: {e}")
                return None
        else:
            return None

    try:
        cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        
        def _monitor_tunnel():
            cf_url = None
            for line in proc.stderr:
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match and not cf_url:
                    cf_url = match.group(0)
                    print("\n" + "=" * 62)
                    print("🌐 CLOUDFLARE HIGH-SPEED PUBLIC TUNNEL READY:")
                    print(f"👉 {cf_url}")
                    print("=" * 62 + "\n")
        
        t = threading.Thread(target=_monitor_tunnel, daemon=True)
        t.start()
        return proc
    except Exception as e:
        logging.warning(f"Failed to start Cloudflare tunnel: {e}")
        return None


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for OmniVoice Studio."""
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create Gradio public link."
    )
    parser.add_argument(
        "--tunnel",
        default="gradio",
        choices=["gradio", "cloudflare", "none"],
        help="Public tunnel provider (gradio, cloudflare, none). Default: gradio.",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:
    """Builds the full Gradio Blocks application for OmniVoice AI Studio."""
    sampling_rate = model.sampling_rate

    # Shared generation core
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
        voice_clone_prompt=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 24),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if voice_clone_prompt is not None:
                kw["voice_clone_prompt"] = voice_clone_prompt
            else:
                if not ref_audio:
                    return None, "Please upload a reference audio."
                kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    _gen = generate_fn if generate_fn is not None else _gen_core

    theme, css = get_theme_and_css()
    head_js = get_head_js()
    storage_badge = "☁️ <b>Lưu trữ: Google Drive</b> <i>(/content/drive/MyDrive/OmniVoice_Studio)</i>" if _IS_GDRIVE else "💻 <b>Lưu trữ: Cục bộ (Local)</b>"

    with gr.Blocks(theme=theme, css=css, head=head_js, title="OmniVoice Studio") as demo:
        with gr.Row(elem_classes="app-header"):
            with gr.Column():
                gr.Markdown(
                    f"""
<h1 class="app-title">⚡ OmniVoice AI Studio</h1>
<p class="app-subtitle">Nền tảng lồng tiếng & Clone Voice kịch bản chuyên nghiệp cho hơn 600+ ngôn ngữ &nbsp;|&nbsp; {storage_badge}</p>
"""
                )

        with gr.Tabs():
            # Build all Tabs modularly (Voice Manager, Voice Clone, Batch Clone, Script Clone, Voice Design, Audio Merger, User Guide)
            vm_comps = build_voice_manager_tab(model, _gen)
            vc_comps = build_voice_clone_tab(_gen)
            bvc_comps = build_batch_clone_tab(model, _gen)
            sc_comps = build_script_clone_tab(model, _gen)
            vd_comps = build_voice_design_tab(_gen)
            am_comps = build_audio_merger_tab()
            guide_comps = build_guide_tab()

        # Cross-tab synchronizer
        def _sync_all_tabs(target_profile=None, msg=""):
            profiles = list_voice_profiles()
            has_profiles = len(profiles) > 0
            selected = target_profile if (target_profile and target_profile in profiles) else (profiles[0] if has_profiles else None)
            
            preview_aud = get_voice_profile_preview(selected) if selected else None
            info_text = format_voice_info(selected)
            
            src_val = "🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if has_profiles else "📤 Tải lên Audio mẫu mới"
            
            preset_grp_vis = gr.update(visible=has_profiles)
            custom_grp_vis = gr.update(visible=not has_profiles)
            src_radio_upd = gr.update(value=src_val)
            dropdown_upd = gr.update(choices=profiles, value=selected)
            preview_upd = gr.update(value=preview_aud)
            
            return (
                # Voice Manager (4):
                dropdown_upd, preview_upd, info_text, msg,
                # Voice Clone (5):
                src_radio_upd, preset_grp_vis, custom_grp_vis, dropdown_upd, preview_upd,
                # Batch Voice Clone (5):
                src_radio_upd, preset_grp_vis, custom_grp_vis, dropdown_upd, preview_upd,
                # Script Clone (5):
                src_radio_upd, preset_grp_vis, custom_grp_vis, dropdown_upd, preview_upd
            )

        def _on_vm_save(name, audio_path, ref_txt, progress=gr.Progress()):
            if not name or not str(name).strip():
                return _sync_all_tabs(None, "❌ Lỗi: Vui lòng đặt tên cho hồ sơ giọng.")
            if not audio_path:
                return _sync_all_tabs(None, "❌ Lỗi: Vui lòng tải lên file âm thanh mẫu.")
            
            try:
                prompt_obj, actual_ref_text = extract_voice_prompt_safely(
                    model=model,
                    audio_path=audio_path,
                    ref_txt=ref_txt,
                    progress_cb=progress
                )
                saved_name = save_voice_profile(
                    name=str(name).strip(),
                    prompt_obj=prompt_obj,
                    metadata={
                        "ref_text": actual_ref_text,
                        "created_at": str(os.path.getmtime(audio_path)) if (isinstance(audio_path, str) and os.path.exists(audio_path)) else "",
                    },
                    preview_audio_path=audio_path,
                )
                progress(1.0, desc="Đã lưu hồ sơ giọng thành công!")
                return _sync_all_tabs(saved_name, f"✅ Đã lưu thành công hồ sơ giọng: '{saved_name}.pt'!")
            except Exception as e:
                import traceback
                traceback.print_exc()
                return _sync_all_tabs(None, f"❌ Lỗi khi trích xuất vector giọng: {e}")

        def _on_vm_refresh():
            return _sync_all_tabs(None, "Đã làm mới danh sách hồ sơ giọng trên toàn bộ hệ thống.")

        def _on_vm_delete(profile_nm):
            if not profile_nm:
                return _sync_all_tabs(None, "Chưa chọn hồ sơ cần xóa.")
            delete_voice_profile(profile_nm)
            return _sync_all_tabs(None, f"Đã xóa hồ sơ: {profile_nm}")

        sync_all_outputs = [
            # Voice Manager (4):
            vm_comps["vm_profiles_list"], vm_comps["vm_preview_audio"], vm_comps["vm_info_md"], vm_comps["vm_status"],
            # Voice Clone (5):
            vc_comps["vc_source_type"], vc_comps["vc_preset_group"], vc_comps["vc_custom_group"], vc_comps["vc_saved_profile"], vc_comps["vc_preset_preview"],
            # Batch Voice Clone (5):
            bvc_comps["bvc_source_type"], bvc_comps["bvc_preset_group"], bvc_comps["bvc_custom_group"], bvc_comps["bvc_saved_profile"], bvc_comps["bvc_preset_preview"],
            # Script Clone (5):
            sc_comps["sc_source_type"], sc_comps["sc_preset_group"], sc_comps["sc_custom_group"], sc_comps["sc_saved_profile"], sc_comps["sc_preset_preview"]
        ]

        vm_comps["vm_save_btn"].click(
            _on_vm_save,
            inputs=[vm_comps["vm_name"], vm_comps["vm_audio"], vm_comps["vm_text"]],
            outputs=sync_all_outputs,
        )
        vm_comps["vm_refresh_btn"].click(
            _on_vm_refresh,
            outputs=sync_all_outputs,
        )
        vm_comps["vm_delete_btn"].click(
            _on_vm_delete,
            inputs=[vm_comps["vm_profiles_list"]],
            outputs=sync_all_outputs,
        )

    return demo


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    if torch.cuda.is_available() and "cuda" in str(device).lower():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("CUDA optimizations enabled: TF32=True, cuDNN benchmark=True")
        except Exception:
            pass

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=dtype,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    # Tunnel & Share logic
    should_share = args.share
    if args.tunnel == "cloudflare":
        start_cloudflare_tunnel(args.port)
        should_share = False
    elif args.tunnel == "gradio":
        should_share = True
    elif args.tunnel == "none":
        should_share = False

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=should_share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
