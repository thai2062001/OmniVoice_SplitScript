import os
import sys
import json
import time
import shutil
import datetime
import soundfile as sf
import numpy as np
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.webui.script_parser import parse_script
from omnivoice.webui.audio_engine import (
    extract_voice_prompt_safely,
    _clean_gpu_memory,
)

TARGET_IDS = [17, 40, 41, 42, 50, 55, 73, 79]

def main():
    script_path = "script.md"
    run_output_dir = "omnivoice/cli/outputs/auto_runs/script_20260910_160542"
    voice_source = "voice.mp3"
    ref_text = "最高の報酬は努力したことによって得られるものだ"
    speed = 0.75
    num_step = 24
    guidance_scale = 2.0
    lang = "Japanese"

    print("📖 Đang nạp kịch bản cập nhật từ script.md cho Nhóm 2...")
    with open(script_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    segments = parse_script(content)
    target_segments = [s for s in segments if s.get("id") in TARGET_IDS]
    print(f"🎯 Số phân đoạn Nhóm 2 cần render lại: {len(target_segments)} (IDs: {TARGET_IDS})")

    device = get_best_device()
    print(f"🖥️ Thiết bị sử dụng: {device}")

    dtype = torch.float32 if str(device).lower() == "cpu" else torch.float16
    print("⏳ Đang nạp mô hình OmniVoice...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )
    print("✅ Đã nạp xong mô hình!")

    print("🎙️ Chuẩn bị vector giọng mẫu...")
    voice_prompt_obj, _ = extract_voice_prompt_safely(
        model=model,
        audio_path=voice_source,
        ref_txt=ref_text,
    )

    gen_config = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=guidance_scale,
        denoise=True,
        preprocess_prompt=True,
        postprocess_output=True,
    )

    for seg in target_segments:
        seg_id = seg.get("id")
        text = seg.get("text", "").strip()
        out_wav_path = os.path.join(run_output_dir, f"segment_{seg_id:04d}.wav")

        print(f"\n▶️ Đang tạo lại phân đoạn #{seg_id:02d}: {text}")
        t0 = time.time()
        audio_out = model.generate(
            text=text,
            language=lang,
            generation_config=gen_config,
            voice_clone_prompt=voice_prompt_obj,
            speed=speed,
        )
        sampling_rate = model.sampling_rate
        waveform = (audio_out[0] * 32767).astype(np.int16)

        # Lưu đè file
        sf.write(out_wav_path, waveform, sampling_rate)
        cost = round(time.time() - t0, 2)
        dur = round(len(waveform) / sampling_rate, 2)
        print(f"   ✅ Hoàn tất #{seg_id:02d} ({dur}s | render: {cost}s) -> {out_wav_path}")
        _clean_gpu_memory()

    print("\n" + "="*70)
    print(f"🎉 Đã hoàn tất render toàn bộ {len(target_segments)} phân đoạn Nhóm 2! (Không thực hiện ghép nối theo yêu cầu)")

if __name__ == "__main__":
    main()
