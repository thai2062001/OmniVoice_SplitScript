import os
import re
import json
import urllib.request
import urllib.error


def analyze_script_with_gemini(script_text: str, api_key: str = "", model_name: str = "gemini-2.5-flash"):
    """
    Analyzes each segment of the script to recommend emotion & voice instruction tags.
    Calls Gemini API using native standard library (no extra pip deps needed).
    """
    key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        raise ValueError("Chưa cung cấp Gemini API Key. Vui lòng nhập API Key vào ô cấu hình hoặc đặt biến môi trường GEMINI_API_KEY.")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"

    prompt = f"""Bạn là chuyên gia phân tích kịch bản lồng tiếng chuyên nghiệp.
Nhiệm vụ của bạn: Đọc từng câu/đoạn trong kịch bản dưới đây. Kịch bản đầu vào có thể là:
- Dạng văn bản thô (mỗi dòng là 1 câu phân đoạn).
- Hoặc dạng phân đoạn đã có sẵn timeline [#1], [#2]...

Hãy phân tích ngữ cảnh của TỪNG PHÂN ĐOẠN/CÂU và chuẩn hóa sang định dạng lồng tiếng sau:
[#1] THỜI GIAN: 0.0 -> <thời_lượng_ước_tính_giây>
VĂN BẢN: <nội dung câu gốc>
CẢM XÚC: <Hài hước / Sôi nổi / Vui vẻ / Nghiêm túc, chỉnh chu / Ngạc nhiên / Thì thầm / Buồn bã / Kịch tính, cao trào / Bình thản...>
HƯỚNG DẪN AI: <Tóm tắt hướng dẫn biểu cảm/nhịp điệu: vd: High energy intro, Steady pace, Whisper secretly, Emphasize key moments, Excited tone...>
------------------------------------------
[#2] THỜI GIAN: <tiếp_tục_cộng_dồn> -> <kết_thúc>
...

Quy tắc quan trọng:
- Ước tính thời lượng (THỜI GIAN) hợp lý theo độ dài câu (khoảng 2.5s đến 6.5s mỗi câu). Nếu kịch bản đã có sẵn thời gian thì giữ nguyên timeline gốc.
- Giữ NGUYÊN VẸN 100% nội dung văn bản từng câu, KHÔNG tự ý chỉnh sửa chữ, không xóa câu, không gộp câu.
- Mỗi câu phải có số thứ tự [#1], [#2], [#3]... liên tục.
- Trả về ĐÚNG định dạng khối phân đoạn như trên, KHÔNG thêm lời chào, giải thích ngoài lề.

--- KỊCH BẢN ĐẦU VÀO ---
{script_text}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
        }
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            candidates = resp_data.get("candidates", [])
            if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
                out_text = candidates[0]["content"]["parts"][0]["text"]
                # Clean markdown backticks if any
                out_text = re.sub(r"^```(?:markdown|text)?\s*", "", out_text.strip(), flags=re.IGNORECASE)
                out_text = re.sub(r"\s*```$", "", out_text.strip())
                return out_text
            else:
                raise ValueError("Không nhận được phản hồi hợp lệ từ Gemini API.")
    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="ignore")
        raise ValueError(f"Gemini API Error ({he.code}): {err_body}")


def map_to_valid_instruct(emotion_str: str, guidance_str: str):
    """
    Maps freeform emotion/guidance keywords from script to valid OmniVoice attributes.
    Ensures only attributes strictly supported by OmniVoice are passed.
    """
    from omnivoice.utils.voice_design import _INSTRUCT_ALL_VALID
    
    combined = f"{emotion_str} {guidance_str}".lower()
    valid_tags = []
    
    # Style
    if any(k in combined for k in ["whisper", "thì thầm", "bí mật", "secret", "tâm sự", "耳语"]):
        valid_tags.append("whisper")
        
    # Pitch / Energy / Emotion
    # High Pitch / Energy: Hài hước, vui tươi, phấn khích, ngạc nhiên, la hét, châm biếm, kịch tính cao
    if any(k in combined for k in ["very high pitch", "hét", "screaming", "cực kỳ phấn khích"]):
        valid_tags.append("very high pitch")
    elif any(k in combined for k in [
        "high energy", "energetic", "excited", "surprised", "high pitch", "cao trào",
        "vui vẻ", "vui tươi", "hài hước", "funny", "humorous", "comedy", "ngạc nhiên",
        "sôi nổi", "hào hứng", "nhiệt huyết", "châm biếm", "dí dỏm", "chúc mừng",
        "cười", "laughing", "tươi tắn"
    ]):
        valid_tags.append("high pitch")
    # Very Low Pitch: Trầm tối, huyền bí, ma mị, giọng đáy
    elif any(k in combined for k in ["very low pitch", "deep voice", "trầm sâu", "ma mị", "u ám", "rùng rợn"]):
        valid_tags.append("very low pitch")
    # Low Pitch: Nghiêm túc, chỉnh chu, tin tức, buồn bã, điềm tĩnh, suy tư, điện ảnh
    elif any(k in combined for k in [
        "low pitch", "serious", "calm", "trầm", "nghiêm túc", "chỉnh chu", "buồn",
        "sad", "sadness", "u buồn", "suy tư", "thất vọng", "lắng đọng", "chững chạc",
        "thuyết minh", "chính luận", "chuyên nghiệp", "tĩnh lặng"
    ]):
        valid_tags.append("low pitch")
    # Moderate Pitch: Bình thường, tự nhiên, tin tức chuẩn, vừa phải
    elif any(k in combined for k in ["informative", "steady pace", "steady", "moderate pitch", "bình thường", "tự nhiên", "chuẩn mực", "ổn định"]):
        valid_tags.append("moderate pitch")
        
    # Age
    if any(k in combined for k in ["child", "kid", "trẻ em", "em bé", "nhí"]):
        valid_tags.append("child")
    elif any(k in combined for k in ["teenager", "teen", "thiếu niên", "học sinh"]):
        valid_tags.append("teenager")
    elif any(k in combined for k in ["elderly", "old", "người già", "ông lão", "bà lão", "lớn tuổi"]):
        valid_tags.append("elderly")
    elif any(k in combined for k in ["young adult", "young", "trẻ", "thanh niên"]):
        valid_tags.append("young adult")
    elif any(k in combined for k in ["middle-aged", "trung niên"]):
        valid_tags.append("middle-aged")
        
    # Gender
    if any(k in combined for k in ["female", "woman", "girl", "nữ", "gái", "chị", "cô"]):
        valid_tags.append("female")
    elif any(k in combined for k in ["male", "man", "boy", "nam", "trai", "anh", "chú"]):
        valid_tags.append("male")
        
    # Accents
    if "japanese" in combined or "tiếng nhật" in combined or "nhật" in combined:
        valid_tags.append("japanese accent")
    elif "british" in combined or "anh" in combined:
        valid_tags.append("british accent")
    elif "american" in combined or "mỹ" in combined:
        valid_tags.append("american accent")
    elif "chinese" in combined or "trung" in combined:
        valid_tags.append("chinese accent")

    filtered = []
    for t in valid_tags:
        if t in _INSTRUCT_ALL_VALID and t not in filtered:
            filtered.append(t)
    return ", ".join(filtered) if filtered else None


def parse_script(script_text: str):
    """
    Parses a script of segments with timelines, text, emotions and instructions.
    Supports both:
    1. Structured blocks with [#ID] THỜI GIAN: X -> Y
    2. Raw line-by-line scripts (each line becomes a segment).
    """
    segments = []
    lines = script_text.split('\n')
    current_seg = None
    has_structured_tags = bool(re.search(r'\[#\d+\]', script_text))
    
    if has_structured_tags:
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            time_match = re.match(r'\[#(\d+)\]\s*THỜI GIAN:\s*([\d\.]+)\s*->\s*([\d\.]+)', line, re.IGNORECASE)
            if time_match:
                if current_seg:
                    segments.append(current_seg)
                seg_id = int(time_match.group(1))
                start = float(time_match.group(2))
                end = float(time_match.group(3))
                current_seg = {
                    "id": seg_id,
                    "duration": round(end - start, 2),
                    "text": "",
                    "emotion": "",
                    "guidance": "",
                }
                continue
                
            if current_seg is None:
                continue
                
            text_match = re.match(r'VĂN BẢN\s*(?:\([^)]+\))?:\s*(.+)', line, re.IGNORECASE)
            if text_match:
                current_seg["text"] = text_match.group(1).strip()
                continue
                
            emotion_match = re.match(r'CẢM XÚC:\s*(.+)', line, re.IGNORECASE)
            if emotion_match:
                current_seg["emotion"] = emotion_match.group(1).strip()
                continue
                
            guidance_match = re.match(r'HƯỚNG DẪN AI:\s*(.+)', line, re.IGNORECASE)
            if guidance_match:
                current_seg["guidance"] = guidance_match.group(1).strip()
                continue
                
        if current_seg:
            segments.append(current_seg)
    else:
        # Fallback: Plain text line-by-line parsing
        seg_counter = 1
        for raw_line in lines:
            cleaned = raw_line.strip()
            if not cleaned or cleaned.startswith("===") or cleaned.startswith("---"):
                continue
            # Estimate reasonable duration based on word count (approx 2.5-3 words per sec)
            words = len(cleaned.split())
            est_duration = max(3.0, round(words * 0.35 + 1.0, 1))
            segments.append({
                "id": seg_counter,
                "duration": est_duration,
                "text": cleaned,
                "emotion": "",
                "guidance": "",
            })
            seg_counter += 1
        
    formatted_segs = []
    for seg in segments:
        instruct_parts = []
        if seg["emotion"]:
            instruct_parts.append(seg["emotion"])
        if seg["guidance"]:
            instruct_parts.append(seg["guidance"])
            
        valid_inst = map_to_valid_instruct(seg["emotion"], seg["guidance"])
            
        formatted_segs.append({
            "id": seg["id"],
            "duration": seg["duration"],
            "text": seg["text"],
            "raw_instruct": ", ".join(instruct_parts),
            "valid_instruct": valid_inst
        })
    return formatted_segs
