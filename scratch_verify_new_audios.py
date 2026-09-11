import os
import sys
import json
import difflib
import unicodedata
import torch
import soundfile as sf
from transformers import pipeline
from omnivoice.webui.script_parser import parse_script

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize('NFKC', text)
    for ch in " 、。！？!?.,「」『』()（）-—… \t\n\r":
        t = t.replace(ch, "")
    return t.lower()

def main():
    script_path = "script.md"
    audio_dir = "omnivoice/cli/outputs/auto_runs/script_20260910_160542"
    re_rendered_ids = [20, 29, 39, 60, 65, 72, 76, 81, 82, 17, 40, 41, 42, 50, 55, 73, 79]

    with open(script_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    segments = parse_script(content)
    seg_dict = {s["id"]: s["text"].strip() for s in segments if "id" in s}

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cpu",
        torch_dtype=torch.float32,
    )

    print("📊 BÁO CÁO KIỂM TRA LẠI CÁC CÂU ĐÃ RENDER:")
    print("=" * 80)

    for seg_id in re_rendered_ids:
        expected_text = seg_dict.get(seg_id, "")
        wav_file = os.path.join(audio_dir, f"segment_{seg_id:04d}.wav")

        info = sf.info(wav_file)
        dur = round(info.duration, 2)

        asr_res = asr_pipe(
            wav_file,
            generate_kwargs={"language": "japanese", "task": "transcribe"}
        )
        transcribed_text = asr_res.get("text", "").strip()

        norm_exp = normalize_text(expected_text)
        norm_trans = normalize_text(transcribed_text)
        matcher = difflib.SequenceMatcher(None, norm_exp, norm_trans)
        similarity = round(matcher.ratio(), 3)

        print(f"[#{seg_id:02d}] ⏱️ {dur}s | Độ khớp: {int(similarity*100)}%")
        print(f"   📝 Kịch bản mới: {expected_text}")
        print(f"   🎙️ Audio nhận diện: {transcribed_text}")
        print("-" * 80)

if __name__ == "__main__":
    main()
