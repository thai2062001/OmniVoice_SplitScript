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
ref_text = "皆さんこんにちは。お元気ですか?この度は私の声を聞いて、使っていただきありがとうございます。皆さんの助けに少しでもなれたら幸いです。"

print("[2/4] Trich xuat vector giong tu voice.mp3 moi...")
prompt, detected_ref = extract_voice_prompt_safely(model, voice_audio, ref_text)

# Render cau 1 voi voice moi
text_test = "デリンクユ地下都市、18層の闇に隠された、古代の巨大要塞。"

gen_config = OmniVoiceGenerationConfig(
    num_step=24,
    guidance_scale=2.0,
    denoise=True,
    preprocess_prompt=True,
    postprocess_output=True,
)

print(f"[3/4] Render test cau 1 voi giong moi: '{text_test}'...")
t0 = time.time()
audio_out = model.generate(
    text=text_test,
    language="Japanese",
    generation_config=gen_config,
    voice_clone_prompt=prompt,
    speed=1.15,
)

waveform = (audio_out[0] * 32767).astype(np.int16)
out_file = "test_new_voice_line1.wav"
sf.write(out_file, waveform, sampling_rate)
dur = round(len(waveform) / sampling_rate, 2)
cost = round(time.time() - t0, 2)

print(f"[4/4] Hoan tat! File: {out_file} ({dur}s | cost: {cost}s)")
