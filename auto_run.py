"""
OmniVoice - 1-Click Auto Run Tool
Chạy tự động sinh giọng theo kịch bản và giọng mẫu thiết lập sẵn, sau đó tự ghép thành file audio hoàn chỉnh.
"""

import os
import sys
import json
import time
import shutil
import argparse
import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("Host Process for Windows Services")
    except Exception:
        pass

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_config.json")


def load_config() -> dict:
    default_cfg = {
        "script_file": "script.md",
        "voice_source": "voice.mp3",
        "ref_text": "",
        "language": "Auto",
        "num_step": 24,
        "guidance_scale": 2.0,
        "speed": 1.0,
        "merge_gap": 0.3,
        "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnivoice", "cli", "outputs", "auto_runs"),
        "use_gemini_analysis": True,
        "auto_merge": True,
        "model_checkpoint": "k2-fsa/OmniVoice",
        "device": None,
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                default_cfg.update(cfg)
        except Exception as e:
            print(f"[CẢNH BÁO] Không đọc được file auto_config.json ({e}), sử dụng thiết lập mặc định.")
    return default_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="OmniVoice 1-Click Auto Run - Tự động nạp kịch bản & lồng tiếng hoàn chỉnh",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", default=CONFIG_FILE, help="Đường dẫn file cấu hình JSON")
    parser.add_argument("--script", default=None, help="Đường dẫn file kịch bản (.txt, .md)")
    parser.add_argument("--voice", default=None, help="Tên hồ sơ giọng (.pt) hoặc đường dẫn file âm thanh mẫu (.wav, .mp3)")
    parser.add_argument("--ref-text", default=None, help="Văn bản mẫu của giọng (tùy chọn)")
    parser.add_argument("--lang", default=None, help="Ngôn ngữ kịch bản (mặc định: Auto)")
    parser.add_argument("--output", default=None, help="Thư mục xuất kết quả")
    parser.add_argument("--steps", type=int, default=None, help="Số bước diffusion (mặc định: 24)")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale (mặc định: 2.0)")
    parser.add_argument("--speed", type=float, default=None, help="Tốc độ đọc (mặc định: 1.0)")
    parser.add_argument("--gap", type=float, default=None, help="Khoảng lặng nghỉ giữa các câu khi ghép (giây)")
    parser.add_argument("--no-gemini", action="store_true", help="Tắt tính năng phân tích kịch bản tự động bằng Gemini")
    parser.add_argument("--no-merge", action="store_true", help="Không ghép các câu lẻ thành file hoàn chỉnh")
    parser.add_argument("--device", default=None, help="Device chạy model (cuda, cpu, ...)")
    return parser.parse_args()


def resolve_file_path(path_str: str) -> str:
    """Tìm đường dẫn file chính xác cả tương đối và tuyệt đối."""
    if not path_str:
        return ""
    clean = path_str.strip().strip('"').strip("'")
    if os.path.isabs(clean) and os.path.exists(clean):
        return clean
    root_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(root_dir, clean)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(clean):
        return os.path.abspath(clean)
    return clean


def main():
    args = parse_args()
    cfg = load_config()

    # Ghi đè cấu hình nếu có tham số CLI
    script_path = resolve_file_path(args.script or cfg.get("script_file", "script.md"))
    voice_source = args.voice or cfg.get("voice_source", "")
    ref_text = args.ref_text if args.ref_text is not None else cfg.get("ref_text", "")
    lang = args.lang or cfg.get("language", "Auto")
    output_base_dir = args.output or cfg.get("output_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnivoice", "cli", "outputs", "auto_runs"))
    num_step = args.steps or int(cfg.get("num_step", 24))
    guidance_scale = args.guidance or float(cfg.get("guidance_scale", 2.0))
    speed = args.speed or float(cfg.get("speed", 1.0))
    merge_gap = args.gap if args.gap is not None else float(cfg.get("merge_gap", 0.3))
    use_gemini = not args.no_gemini if args.no_gemini else bool(cfg.get("use_gemini_analysis", True))
    auto_merge = not args.no_merge if args.no_merge else bool(cfg.get("auto_merge", True))
    checkpoint = cfg.get("model_checkpoint", "k2-fsa/OmniVoice")

    print("\n" + "=" * 65)
    print("🚀 OMNIVOICE 1-CLICK AUTO RUNNER")
    print("=" * 65)
    print(f"📄 Kịch bản       : {script_path}")
    print(f"🎙️ Giọng mẫu       : {voice_source}")
    print(f"🌐 Ngôn ngữ       : {lang}")
    print(f"⚙️ Tham số        : Steps={num_step}, Guidance={guidance_scale}, Speed={speed}")
    print(f"🧩 Tự động ghép   : {'Bật' if auto_merge else 'Tắt'} (Khoảng nghỉ: {merge_gap}s)")
    print(f"🤖 Phân tích AI   : {'Gemini AI' if use_gemini else 'Tắt'}")
    print("=" * 65 + "\n")

    # 1. Kiểm tra kịch bản
    if not os.path.exists(script_path):
        print(f"❌ [LỖI] Không tìm thấy file kịch bản tại: {script_path}")
        print("💡 Vui lòng chỉnh lại 'script_file' trong 'auto_config.json' hoặc truyền cờ '--script <duong_dan>'")
        sys.exit(1)

    with open(script_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        raw_script_content = f.read().strip()

    if not raw_script_content:
        print(f"❌ [LỖI] File kịch bản rỗng: {script_path}")
        sys.exit(1)

    # 2. Phân tích kịch bản
    from omnivoice.webui.script_parser import parse_script, analyze_script_with_gemini
    has_structured_tags = "[#" in raw_script_content and "THỜI GIAN:" in raw_script_content
    if not has_structured_tags and use_gemini:
        print("🔍 Đang sử dụng Gemini AI để phân tích cảm xúc & timeline kịch bản...")
        try:
            analyzed_text = analyze_script_with_gemini(raw_script_content)
            print("✅ Đã phân tích kịch bản thành công qua Gemini AI.")
            script_content = analyzed_text
        except Exception as ge:
            print(f"⚠️ [CẢNH BÁO] Không thể gọi Gemini ({ge}). Sẽ chuyển sang tách câu trực tiếp.")
            script_content = raw_script_content
    else:
        script_content = raw_script_content

    segments = parse_script(script_content)
    if not segments:
        print("❌ [LỖI] Không tìm thấy phân đoạn hợp lệ nào trong kịch bản.")
        sys.exit(1)

    total_words = sum(len(s.get("text", "").split()) for s in segments)
    print(f"📊 Tổng số phân đoạn: {len(segments)} câu | Tổng số từ: {total_words} từ\n")

    # 3. Tải các thư viện AI nặng và mô hình OmniVoice
    import torch
    import numpy as np
    import soundfile as sf
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    from omnivoice.utils.common import get_best_device
    from omnivoice.webui.profile_manager import list_voice_profiles, load_voice_profile
    from omnivoice.webui.audio_engine import (
        extract_voice_prompt_safely,
        process_audio_merger,
        _clean_gpu_memory,
    )

    device = args.device or cfg.get("device") or get_best_device()

    print(f"⏳ Đang khởi tạo mô hình OmniVoice trên thiết bị [{device}]...")
    if torch.cuda.is_available() and "cuda" in str(device).lower():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    dtype = torch.float32 if str(device).lower() == "cpu" else torch.float16
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=dtype,
        load_asr=True,
    )
    sampling_rate = model.sampling_rate
    print("✅ Đã nạp xong mô hình OmniVoice!")

    # 4. Chuẩn bị giọng mẫu (Voice Clone Prompt)
    print("⏳ Đang chuẩn bị dữ liệu giọng mẫu...")
    saved_profiles = list_voice_profiles()
    voice_prompt_obj = None

    clean_v = voice_source.strip().replace(".pt", "")
    if clean_v in saved_profiles:
        print(f"🎙️ Nạp Hồ sơ giọng đã lưu: [{clean_v}]")
        voice_prompt_obj = load_voice_profile(clean_v)
    else:
        local_audio_path = resolve_file_path(voice_source)
        if local_audio_path and os.path.exists(local_audio_path):
            print(f"🎙️ Trích xuất vector giọng từ file: [{local_audio_path}]")
            voice_prompt_obj, detected_ref_text = extract_voice_prompt_safely(
                model=model,
                audio_path=local_audio_path,
                ref_txt=ref_text,
            )
            print(f"📝 Văn bản tham chiếu: {detected_ref_text}")
        elif saved_profiles:
            fallback_prof = saved_profiles[0]
            print(f"⚠️ Không tìm thấy file '{voice_source}'. Tự động dùng hồ sơ đầu tiên: [{fallback_prof}]")
            voice_prompt_obj = load_voice_profile(fallback_prof)
        else:
            print(f"❌ [LỖI] Không tìm thấy file âm thanh hoặc hồ sơ giọng: '{voice_source}'")
            sys.exit(1)

    # 5. Tìm hoặc tạo thư mục lưu kết quả (Hỗ trợ Resume tiếp tục phiên cũ)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    existing_dirs = []
    if os.path.exists(output_base_dir):
        for d in os.listdir(output_base_dir):
            full_d = os.path.join(output_base_dir, d)
            if os.path.isdir(full_d) and d.startswith(f"{script_name}_"):
                existing_dirs.append(full_d)
    
    existing_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if existing_dirs and any(f.startswith("segment_") for f in os.listdir(existing_dirs[0])):
        run_output_dir = existing_dirs[0]
        print(f"🔄 [RESUME] Phát hiện phiên chạy trước đó tại: {run_output_dir}")
        print("💡 Hệ thống sẽ tự động BỎ QUA các câu đã tạo và TIẾP TỤC các câu còn lại!\n")
    else:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_base_dir, f"{script_name}_{timestamp_str}")
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"📁 Thư mục lưu kết quả: {run_output_dir}\n")

    with open(os.path.join(run_output_dir, "parsed_script.txt"), "w", encoding="utf-8") as f:
        f.write(script_content)

    # 6. Tiến hành sinh giọng từng phân đoạn
    print("▶️ BẮT ĐẦU TIẾN TRÌNH SINH GIỌNG...")
    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=guidance_scale,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    lang_param = lang if (lang and lang != "Auto") else None
    generated_files = []
    start_all_time = time.time()

    for idx, seg in enumerate(segments, 1):
        seg_id = seg.get("id", idx)
        text = seg.get("text", "").strip()
        instruct = seg.get("valid_instruct") or seg.get("raw_instruct") or None
        target_dur = seg.get("duration")

        out_wav_name = f"segment_{seg_id:04d}.wav"
        out_wav_path = os.path.join(run_output_dir, out_wav_name)

        # Hỗ trợ Resume: Nếu câu đã được tạo trước đó thì bỏ qua không render lại
        if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 1000:
            try:
                info = sf.info(out_wav_path)
                print(f"[{idx}/{len(segments)}] Phân đoạn #{seg_id}: {text[:45]}... ⏭️ [ĐÃ CÓ SẴN ({round(info.duration, 2)}s) - BỎ QUA]")
                generated_files.append(out_wav_path)
                continue
            except Exception:
                pass

        print(f"[{idx}/{len(segments)}] Phân đoạn #{seg_id}: {text[:50]}... ", end="", flush=True)
        seg_start = time.time()

        kw = {
            "text": text,
            "language": lang_param,
            "generation_config": gen_config,
            "voice_clone_prompt": voice_prompt_obj,
        }
        if speed != 1.0:
            kw["speed"] = speed
        if has_structured_tags and target_dur and float(target_dur) > 0:
            kw["duration"] = float(target_dur)
        if instruct:
            kw["instruct"] = instruct

        try:
            audio_out = model.generate(**kw)
            waveform = (audio_out[0] * 32767).astype(np.int16)
            sf.write(out_wav_path, waveform, sampling_rate)
            duration_sec = round(len(waveform) / sampling_rate, 2)
            gen_cost = round(time.time() - seg_start, 2)
            print(f"✅ ({duration_sec}s | render: {gen_cost}s)")
            generated_files.append(out_wav_path)
        except Exception as gen_err:
            print(f"❌ Lỗi: {gen_err}")

        _clean_gpu_memory()

    total_cost = round(time.time() - start_all_time, 2)
    print(f"\n🎉 Đã hoàn thành sinh giọng cho {len(generated_files)}/{len(segments)} phân đoạn! (Tổng thời gian: {total_cost}s)")

    # 7. Tự động ghép toàn bộ các file thành 1 file duy nhất
    if auto_merge and generated_files:
        print(f"\n🧩 Đang ghép nối toàn bộ {len(generated_files)} file thành 1 file Audio hoàn chỉnh...")
        try:
            status_msg, merged_filepath, _ = process_audio_merger(
                mode="Quét thư mục cục bộ (Nhập đường dẫn)",
                folder_path=run_output_dir,
                uploaded_files=None,
                gap_sec=merge_gap,
            )
            if merged_filepath and os.path.exists(merged_filepath):
                cur_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                final_merged_name = f"FINAL_MERGED_{script_name}_{cur_time_str}.wav"
                final_merged_path = os.path.join(run_output_dir, final_merged_name)
                shutil.copy2(merged_filepath, final_merged_path)
                print("=" * 65)
                print("🎉 AUDIO HOÀN CHỈNH ĐÃ ĐƯỢC TẠO THÀNH CÔNG:")
                print(f"👉 {final_merged_path}")
                print("=" * 65)
            else:
                print(f"⚠️ Trạng thái ghép: {status_msg}")
        except Exception as me:
            print(f"⚠️ Lỗi khi ghép file audio: {me}")

    print(f"\n📂 Tất cả file kết quả được lưu tại: {os.path.abspath(run_output_dir)}")


if __name__ == "__main__":
    main()
