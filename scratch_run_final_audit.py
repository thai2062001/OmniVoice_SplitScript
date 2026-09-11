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

    with open(script_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    segments = parse_script(content)
    print(f"Bắt đầu quét toàn bộ {len(segments)} file...")

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cpu",
        torch_dtype=torch.float32,
    )

    results = []

    for idx, seg in enumerate(segments, 1):
        seg_id = seg.get("id", idx)
        expected_text = seg.get("text", "").strip()
        wav_file = os.path.join(audio_dir, f"segment_{seg_id:04d}.wav")

        if not os.path.exists(wav_file):
            results.append({
                "id": seg_id,
                "status": "MISSING",
                "expected": expected_text,
                "transcribed": "",
                "duration": 0,
                "similarity": 0,
                "issues": ["Thiếu file audio"]
            })
            continue

        info = sf.info(wav_file)
        dur = round(info.duration, 2)

        res = asr_pipe(wav_file, generate_kwargs={"language": "japanese", "task": "transcribe"})
        trans_text = res.get("text", "").strip()

        norm_exp = normalize_text(expected_text)
        norm_trans = normalize_text(trans_text)
        sim = round(difflib.SequenceMatcher(None, norm_exp, norm_trans).ratio(), 3)

        issues = []
        if dur < 1.0:
            issues.append("Thời lượng quá ngắn")

        # Kiểm tra lặp từ ngữ
        words = trans_text.split()
        for w in words:
            if len(w) >= 3 and trans_text.count(w) >= 3 and expected_text.count(w) < 2:
                issues.append(f"Lặp từ: {w}")
                break

        # Check nếu độ khớp thấp
        if sim < 0.70:
            issues.append(f"Độ lệch cao ({int(sim*100)}%)")
        elif sim < 0.85:
            issues.append(f"Độ lệch vừa ({int(sim*100)}%)")

        results.append({
            "id": seg_id,
            "status": "WARNING" if issues else "OK",
            "expected": expected_text,
            "transcribed": trans_text,
            "duration": dur,
            "similarity": sim,
            "issues": issues
        })

    out_json = os.path.join(audio_dir, "audit_final_all_91.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Hoàn tất quét toàn bộ 91 file!")

if __name__ == "__main__":
    main()
