import json

r = json.load(open('omnivoice/cli/outputs/auto_runs/script_20260910_160542/check_report.json', encoding='utf-8'))

severe = [] # Độ khớp < 75% (có khả năng nuốt câu, sai từ nghiêm trọng, nói lắp, đọc bậy)
moderate = [] # Độ khớp 75% - 85% (sai lệch từ vựng / kanji kana)

for x in r:
    sim = x.get("similarity", 0)
    if sim < 0.75:
        severe.append(x)
    elif sim < 0.85:
        moderate.append(x)

with open('scratch_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(f"=== DANH SÁCH LỖI NẶNG (SIMILARITY < 75%) : {len(severe)} câu ===\n\n")
    for s in severe:
        f.write(f"[#{s['id']:02d}] (Độ khớp: {int(s['similarity']*100)}% | Thời lượng: {s['duration']}s)\n")
        f.write(f"  - Gốc: {s['expected']}\n")
        f.write(f"  - ASR: {s['transcribed']}\n")
        f.write(f"  - Vấn đề: {', '.join(s['issues'])}\n\n")

    f.write(f"\n=== DANH SÁCH CẦN LƯU Ý (SIMILARITY 75% - 85%) : {len(moderate)} câu ===\n\n")
    for m in moderate:
        f.write(f"[#{m['id']:02d}] (Độ khớp: {int(m['similarity']*100)}% | Thời lượng: {m['duration']}s)\n")
        f.write(f"  - Gốc: {m['expected']}\n")
        f.write(f"  - ASR: {m['transcribed']}\n")
        f.write(f"  - Vấn đề: {', '.join(m['issues'])}\n\n")

print(f"Severe: {len(severe)}, Moderate: {len(moderate)}")
