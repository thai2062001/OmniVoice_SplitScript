import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import numpy as np
import soundfile as sf
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.webui.audio_engine import extract_voice_prompt_safely

device = "cpu"
checkpoint = "k2-fsa/OmniVoice"

print(f"[1/4] Nap model OmniVoice tren {device}...")
model = OmniVoice.from_pretrained(checkpoint, device_map=device, dtype=torch.float32, load_asr=True)
sampling_rate = model.sampling_rate

voice_audio = "voice.mp3"
# Cập nhật chuẩn 100% nội dung thực tế của voice.mp3
ref_text = "こんにちは 私の声は日本語ナレーション セリフ ポッドキャストやショートムービーで使われています いろんなところに使ってみてくださいね"

print("[2/4] Trich xuat vector giong tu voice.mp3 voi ref_text chuan...")
prompt, detected_ref = extract_voice_prompt_safely(model, voice_audio, ref_text)

# Văn bản câu 1 đã làm sạch hoàn toàn ký tự markdown, bắt đầu thẳng vào nội dung
text_1 = "デリンクユ地下都市、18層の闇に隠された古代の超巨大生存要塞。トルコの荒涼とした大地の地下85メートル。そこには、2万人以上の命をかくすに足る超巨大地下都市が存在しています。そして、その驚異の扉が開かれたのは、一本のハンマーによる偶然の一撃でした。"

speed_val = 1.15
print(f"[3/4] Dang render file mau 1 voi speed = {speed_val} (Tối ưu phát âm, khử hallucination)...")
gen_config = OmniVoiceGenerationConfig(
    num_step=24,
    guidance_scale=2.0,
    denoise=True,
    preprocess_prompt=True,
    postprocess_output=True,
)

t0 = time.time()
audio_out = model.generate(
    text=text_1,
    language="Japanese",
    generation_config=gen_config,
    voice_clone_prompt=prompt,
    speed=speed_val,
)

waveform = (audio_out[0] * 32767).astype(np.int16)
out_file = "test_sample_segment_1_speed115_clean.wav"
sf.write(out_file, waveform, sampling_rate)

dur = round(len(waveform) / sampling_rate, 2)
cost = round(time.time() - t0, 2)
print(f"[4/4] Hoan tat! File: {out_file} ({dur}s | render: {cost}s)")
