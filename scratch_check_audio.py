import os
import sys
import json
import time
import difflib
import unicodedata
import torch
import soundfile as sf
from omnivoice import OmniVoice
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
    # Normalize unicode
    t = unicodedata.normalize('NFKC', text)
    # Remove punctuation, spaces, hiragana/katakana/kanji normalization
    # Remove Japanese and English punctuation
    for ch in " 、。！？!?.,「」『』()（）-—… \t\n\r":
        t = t.replace(ch, "")
    return t.lower()

def main():
    script_path = "script.md"
    audio_dir = "omnivoice/cli/outputs/auto_runs/script_20260910_160542"

    print("📖 Đang đọc kịch bản gốc...")
    with open(script_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    segments = parse_script(content)
    print(f"📊 Tổng số phân đoạn cần kiểm tra: {len(segments)}")

    print("🧠 Đang tải mô hình Whisper ASR tích hợp trong OmniVoice...")
    # Khởi tạo OmniVoice chỉ để dùng transcribe ASR
    # OmniVoice.load_asr_model
    from transformers import pipeline
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cpu",
        torch_dtype=torch.float32,
    )
    print("✅ ASR Model sẵn sàng!")

    results = []
    print("\n🔍 BẮT ĐẦU KIỂM TRA CHI TIẾT TỪNG FILE AUDIO...")
    print("=" * 80)

    for idx, seg in enumerate(segments, 1):
        seg_id = seg.get("id", idx)
        expected_text = seg.get("text", "").strip()
        wav_file = os.path.join(audio_dir, f"segment_{seg_id:04d}.wav")

        if not os.path.exists(wav_file):
            print(f"[#{seg_id:02d}] ❌ KHÔNG TÌM THẤY FILE: {wav_file}")
            results.append({
                "id": seg_id,
                "status": "MISSING",
                "expected": expected_text,
                "transcribed": "",
                "duration": 0,
                "issues": ["Thiếu file audio"],
                "similarity": 0.0
            })
            continue

        try:
            info = sf.info(wav_file)
            dur = round(info.duration, 2)

            # Transcribe audio
            asr_res = asr_pipe(
                wav_file,
                generate_kwargs={"language": "japanese", "task": "transcribe"}
            )
            transcribed_text = asr_res.get("text", "").strip()

            # Normalize for comparison
            norm_exp = normalize_text(expected_text)
            norm_trans = normalize_text(transcribed_text)

            # Similarity ratio
            matcher = difflib.SequenceMatcher(None, norm_exp, norm_trans)
            similarity = round(matcher.ratio(), 3)

            issues = []
            
            # Check length / duration anomaly
            if dur < 1.0:
                issues.append("Thời lượng quá ngắn (< 1s)")
            
            # Check repetition
            # Detect repeated phrases in transcribed text
            words = transcribed_text.split()
            for w in words:
                if len(w) > 3 and transcribed_text.count(w) >= 3 and expected_text.count(w) < 2:
                    issues.append(f"Nghi vấn lặp từ: '{w}'")
                    break

            # Compare match
            if similarity < 0.75:
                issues.append(f"Độ khớp thấp ({int(similarity*100)}%) - Có thể sai âm / thiếu chữ / thừa âm")
            elif similarity < 0.90:
                issues.append(f"Khớp tương đối ({int(similarity*100)}%) - Có sai lệch nhỏ")

            status = "WARNING" if issues else "OK"

            results.append({
                "id": seg_id,
                "status": status,
                "expected": expected_text,
                "transcribed": transcribed_text,
                "duration": dur,
                "issues": issues,
                "similarity": similarity
            })

            status_icon = "✅" if status == "OK" else "⚠️"
            print(f"[#{seg_id:02d}] {status_icon} ({dur}s | Khớp: {int(similarity*100)}%)")
            print(f"   Gốc: {expected_text}")
            print(f"   ASR: {transcribed_text}")
            if issues:
                print(f"   👉 Vấn đề: {', '.join(issues)}")
            print("-" * 80)

        except Exception as e:
            print(f"[#{seg_id:02d}] ❌ LỖI KHI CHECK: {e}")
            results.append({
                "id": seg_id,
                "status": "ERROR",
                "expected": expected_text,
                "transcribed": "",
                "duration": 0,
                "issues": [f"Lỗi: {str(e)}"],
                "similarity": 0.0
            })

    # Lưu báo cáo dạng JSON
    report_file = os.path.join(audio_dir, "check_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Báo cáo JSON chi tiết đã được lưu tại: {report_file}")

if __name__ == "__main__":
    main()
