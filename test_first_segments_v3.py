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

# V3: Tối ưu triệt để ngữ âm, ngắt câu và từ vựng
test_segments_v3 = [
    # Câu 1: '古代の、きょだいな生存要塞です' (rõ ràng, không bị dính chữ siêu khổng lồ, phát âm chuẩn 100%)
    ("test_seg_1_v3.wav", "デリンクユ地下都市。それは、じゅうはっそうの闇に隠された、古代の、きょだいな生存要塞です。"),
    # Câu 2: '地下、はちじゅうごメートル' (tách dấu phẩy ngắt nhịp để không bị đọc thành 'chijuugometoru')
    ("test_seg_2_v3.wav", "トルコの荒涼とした大地の、地下、はちじゅうごメートル。"),
    # Câu 3: Chuẩn 100%
    ("test_seg_3_v3.wav", "そこには、にまんにん以上の命をかくすことのできる、巨大な地下都市が存在しています。"),
    # Câu 4: 'その驚異の扉が開かれたのは' (viết 開かれた rõ ràng, không nuốt âm)
    ("test_seg_4_v3.wav", "そして、その驚異の扉が、開かれたのは、まさに一本のハンマーによる、偶然の一撃でした。"),
]

gen_config = OmniVoiceGenerationConfig(
    num_step=24,
    guidance_scale=2.0,
    denoise=True,
    preprocess_prompt=True,
    postprocess_output=True,
)

print("[3/4] Render test 4 cau ban V3...")
for fname, text in test_segments_v3:
    t0 = time.time()
    audio_out = model.generate(
        text=text,
        language="Japanese",
        generation_config=gen_config,
        voice_clone_prompt=prompt,
        speed=1.10,
    )
    waveform = (audio_out[0] * 32767).astype(np.int16)
    sf.write(fname, waveform, sampling_rate)
    dur = round(len(waveform) / sampling_rate, 2)
    cost = round(time.time() - t0, 2)
    print(f"✅ Rendered {fname}: '{text[:35]}...' ({dur}s | cost: {cost}s)")

print("[4/4] Hoan tat test render v3!")
