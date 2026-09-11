import json

r = json.load(open('omnivoice/cli/outputs/auto_runs/script_20260910_160542/full_audit_report_v2.json', encoding='utf-8'))

# Lọc các câu có dấu hiệu bất thường thực sự (lặp từ, cụt câu, sai từ rõ rệt)
suspicious = []
for item in r:
    issues = item.get("issues", [])
    sim = item.get("similarity", 0)
    dur = item.get("duration", 0)
    exp = item.get("expected", "")
    trans = item.get("transcribed", "")
    
    # Check if there is repetition or severe mismatch (< 75%)
    if sim < 0.75 or any("lặp" in iss.lower() for iss in issues):
        suspicious.append(item)

print(f"Tong so cau can xem xet: {len(suspicious)}")
with open('scratch_final_audit_summary.txt', 'w', encoding='utf-8') as f:
    for s in suspicious:
        f.write(f"[#{s['id']:02d}] (Độ khớp: {int(s['similarity']*100)}% | {s['duration']}s)\n")
        f.write(f"  - Kịch bản: {s['expected']}\n")
        f.write(f"  - Audio:    {s['transcribed']}\n")
        f.write(f"  - Vấn đề:   {', '.join(s['issues'])}\n\n")
