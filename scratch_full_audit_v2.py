import os
import sys
import json
import time
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

    print("📖 Đang nạp kịch bản script.md hiện tại...")
    with open(script_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    segments = parse_script(content)
    print(f"📊 Tổng số phân đoạn cần quét kiểm tra: {len(segments)}")

    print("🧠 Đang khởi động pipeline Whisper ASR...")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cpu",
        torch_dtype=torch.float32,
    )
    print("✅ Whisper ASR đã sẵn sàng!\n")

    results = []
    print("🔍 TIẾN TRÌNH QUÉT KIỂM TRA TOÀN DIỆN 91 FILE AUDIO:")
    print("=" * 85)

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
                "similarity": 0.0,
                "issues": ["Thiếu file audio"]
            })
            print(f"[#{seg_id:02d}] ❌ THIẾU FILE AUDIO: {wav_file}")
            continue

        try:
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

            issues = []

            # Kiểm tra thời lượng bất thường
            if dur < 1.0:
                issues.append("Thời lượng quá ngắn (< 1s)")

            # Kiểm tra lặp từ ngữ
            words = transcribed_text.split()
            for w in words:
                if len(w) > 3 and transcribed_text.count(w) >= 3 and expected_text.count(w) < 2:
                    issues.append(f"Nghi vấn lặp từ: '{w}'")
                    break

            # Phân loại độ khớp
            if similarity < 0.70:
                issues.append(f"Độ lệch cao ({int(similarity*100)}%) - Có thể sai âm / thiếu chữ / đọc khác kịch bản")
            elif similarity < 0.85:
                issues.append(f"Độ lệch vừa ({int(similarity*100)}%) - Cần kiểm tra đối chiếu từ vựng")

            status = "WARNING" if issues else "OK"

            results.append({
                "id": seg_id,
                "status": status,
                "expected": expected_text,
                "transcribed": transcribed_text,
                "duration": dur,
                "similarity": similarity,
                "issues": issues
            })

            icon = "✅" if status == "OK" else "⚠️"
            print(f"[#{seg_id:02d}] {icon} ({dur}s | Độ khớp: {int(similarity*100)}%)")
            print(f"   📝 Gốc: {expected_text}")
            print(f"   🎙️ ASR: {transcribed_text}")
            if issues:
                print(f"   👉 Chú ý: {', '.join(issues)}")
            print("-" * 85)

        except Exception as e:
            results.append({
                "id": seg_id,
                "status": "ERROR",
                "expected": expected_text,
                "transcribed": "",
                "duration": 0,
                "similarity": 0.0,
                "issues": [f"Lỗi: {str(e)}"]
            })
            print(f"[#{seg_id:02d}] ❌ LỖI: {e}")

    # Lưu báo cáo JSON
    report_file = os.path.join(audio_dir, "full_audit_report_v2.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Đã xuất báo cáo toàn diện tại: {report_file}")

if __name__ == "__main__":
    main()
