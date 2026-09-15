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
ref_text = "こんにちは 私の声は日本語ナレーション セリフ ポッドキャストやショートムービーで使われています いろんなところに使ってみてくださいね"

print("[2/4] Trich xuat vector giong tu voice.mp3 voi ref_text chuan...")
prompt, detected_ref = extract_voice_prompt_safely(model, voice_audio, ref_text)

# Đoạn 1 mới (cực ngắn, tự nhiên, mở màn kịch tính)
test_segments = [
    ("test_seg_1.wav", "デリンクユ地下都市。それは、18層の闇に隠された、古代の超巨大生存要塞です。"),
    ("test_seg_2.wav", "トルコの荒涼とした大地の地下85メートル。"),
    ("test_seg_3.wav", "そこには、2万人以上の命をかくすことのできる、超巨大な地下都市が存在しています。"),
    ("test_seg_4.wav", "そして、その驚異の扉が開かれたのは、まさに一本のハンマーによる、偶然の一撃でした。"),
]

gen_config = OmniVoiceGenerationConfig(
    num_step=24,
    guidance_scale=2.0,
    denoise=True,
    preprocess_prompt=True,
    postprocess_output=True,
)

print("[3/4] Render test 4 cau mo man voi speed 1.15...")
for fname, text in test_segments:
    t0 = time.time()
    audio_out = model.generate(
        text=text,
        language="Japanese",
        generation_config=gen_config,
        voice_clone_prompt=prompt,
        speed=1.15,
    )
    waveform = (audio_out[0] * 32767).astype(np.int16)
    sf.write(fname, waveform, sampling_rate)
    dur = round(len(waveform) / sampling_rate, 2)
    cost = round(time.time() - t0, 2)
    print(f"✅ Rendered {fname}: '{text[:30]}...' ({dur}s | cost: {cost}s)")

print("[4/4] Hoan tat test render!")
