#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import os
import re
import tempfile
import zipfile
import subprocess
import shutil
import time
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

import json
import urllib.request
import urllib.error
import gc

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


def _clean_gpu_memory():
    """Safely frees cached CUDA VRAM and forces garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Storage Directory Detection (Auto-detect Google Drive vs Local)
# ---------------------------------------------------------------------------
def _get_storage_dirs():
    """
    Detects if running on Google Colab with Google Drive mounted.
    Returns (saved_voices_dir, outputs_dir, is_gdrive).
    """
    gdrive_base = "/content/drive/MyDrive/OmniVoice_Studio"
    if os.path.exists("/content/drive/MyDrive"):
        voices_dir = os.path.join(gdrive_base, "saved_voices")
        out_dir = os.path.join(gdrive_base, "outputs")
        is_gdrive = True
    else:
        # Fallback to local workspace
        local_base = os.path.dirname(os.path.abspath(__file__))
        voices_dir = os.path.join(local_base, "saved_voices")
        out_dir = os.path.join(local_base, "outputs")
        is_gdrive = False

    os.makedirs(voices_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    return voices_dir, out_dir, is_gdrive

_SAVED_VOICES_DIR, _OUTPUTS_DIR, _IS_GDRIVE = _get_storage_dirs()
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
# ---------------------------------------------------------------------------
# Saved Voice Profiles Management (.pt embedding storage)
# ---------------------------------------------------------------------------


def list_voice_profiles():
    """Lists all saved voice profile names."""
    if not os.path.exists(_SAVED_VOICES_DIR):
        return []
    profiles = []
    for f in os.listdir(_SAVED_VOICES_DIR):
        if f.endswith(".pt"):
            profiles.append(os.path.splitext(f)[0])
    return sorted(profiles)


def get_voice_profile_metadata(name: str):
    """Loads metadata dictionary for a given voice profile."""
    if not name:
        return {}
    filepath = os.path.join(_SAVED_VOICES_DIR, f"{name}.pt")
    if not os.path.exists(filepath):
        return {}
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            meta = data.get("metadata", {})
            if "ref_text" in data and not meta.get("ref_text"):
                meta["ref_text"] = data.get("ref_text", "")
            return meta
    except Exception as e:
        logging.warning(f"Failed to read metadata for {name}: {e}")
    return {}


def _slugify_voice_name(name: str) -> str:
    """Safely converts voice profile name to clean filesystem-safe ASCII filename."""
    import unicodedata
    raw = (name or "").strip()
    raw = raw.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize('NFKD', raw)
    ascii_text = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    safe = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', ascii_text)
    safe = re.sub(r'[_\-]+', '_', safe).strip('_.')
    if not safe:
        safe = f"voice_profile_{int(time.time())}"
    return safe


def save_voice_profile(name: str, prompt_obj, metadata: dict = None, preview_audio_path: str = None):
    """Saves encoded voice prompt and metadata into .pt file."""
    raw = (name or "").strip()
    safe_name = _slugify_voice_name(raw)
        
    os.makedirs(_SAVED_VOICES_DIR, exist_ok=True)
    filepath = os.path.join(_SAVED_VOICES_DIR, f"{safe_name}.pt")
    
    saved_preview = None
    if preview_audio_path and os.path.exists(preview_audio_path):
        try:
            preview_ext = os.path.splitext(preview_audio_path)[1] or ".wav"
            saved_preview = os.path.join(_SAVED_VOICES_DIR, f"{safe_name}{preview_ext}")
            if os.path.abspath(preview_audio_path) != os.path.abspath(saved_preview):
                shutil.copy2(preview_audio_path, saved_preview)
        except Exception as e:
            logging.warning(f"Could not copy preview audio: {e}")

    meta = metadata or {}
    meta["display_name"] = raw
    meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if hasattr(prompt_obj, "ref_audio_tokens"):
        tokens = prompt_obj.ref_audio_tokens
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu()
        ref_text_val = getattr(prompt_obj, "ref_text", "") or meta.get("ref_text", "")
        ref_rms_val = float(getattr(prompt_obj, "ref_rms", 0.0))
        data = {
            "format_version": 1,
            "ref_audio_tokens": tokens,
            "ref_text": ref_text_val,
            "ref_rms": ref_rms_val,
            "metadata": meta,
            "preview_audio": saved_preview,
        }
    elif isinstance(prompt_obj, dict):
        data = {
            "format_version": 1,
            "ref_audio_tokens": prompt_obj.get("ref_audio_tokens"),
            "ref_text": prompt_obj.get("ref_text", "") or meta.get("ref_text", ""),
            "ref_rms": float(prompt_obj.get("ref_rms", 0.0)),
            "metadata": meta,
            "preview_audio": saved_preview,
        }
    else:
        data = {
            "prompt": prompt_obj,
            "metadata": meta,
            "preview_audio": saved_preview,
        }
        
    torch.save(data, filepath)
    return safe_name


def load_voice_profile(name: str):
    """Loads voice prompt tensor/object and metadata from .pt file."""
    if not name:
        return None, {}
    filepath = os.path.join(_SAVED_VOICES_DIR, f"{name}.pt")
    if not os.path.exists(filepath):
        return None, {}
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if "ref_audio_tokens" in data and data["ref_audio_tokens"] is not None:
                from omnivoice.models.omnivoice import VoiceClonePrompt
                prompt = VoiceClonePrompt(
                    ref_audio_tokens=data["ref_audio_tokens"],
                    ref_text=data.get("ref_text", "") or "",
                    ref_rms=float(data.get("ref_rms", 0.0)),
                )
                return prompt, data.get("metadata", {})
            return data.get("prompt"), data.get("metadata", {})
        return data, {}
    except Exception as e:
        logging.error(f"Error loading voice profile {name}: {e}")
        return None, {}


def get_voice_profile_preview(name: str):
    """Gets preview audio path if saved."""
    if not name:
        return None
    filepath = os.path.join(_SAVED_VOICES_DIR, f"{name}.pt")
    if not os.path.exists(filepath):
        return None
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "preview_audio" in data and data["preview_audio"]:
            p = data["preview_audio"]
            if os.path.exists(p):
                return p
            fallback_local = os.path.join(_SAVED_VOICES_DIR, os.path.basename(p))
            if os.path.exists(fallback_local):
                return fallback_local
    except Exception:
        pass
        
    # Fallback: check for direct audio files with same name
    for ext in [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm"]:
        aud = os.path.join(_SAVED_VOICES_DIR, f"{name}{ext}")
        if os.path.exists(aud):
            return aud
    return None


def delete_voice_profile(name: str):
    """Deletes a saved voice profile and its preview audio."""
    if not name:
        return False
    pt_path = os.path.join(_SAVED_VOICES_DIR, f"{name}.pt")
    preview_to_delete = None
    if os.path.exists(pt_path):
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "preview_audio" in data:
                preview_to_delete = data["preview_audio"]
            os.remove(pt_path)
        except Exception:
            pass

    if preview_to_delete and os.path.exists(preview_to_delete):
        try:
            os.remove(preview_to_delete)
        except Exception:
            pass

    for ext in [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm"]:
        aud = os.path.join(_SAVED_VOICES_DIR, f"{name}{ext}")
        if os.path.exists(aud):
            try:
                os.remove(aud)
            except Exception:
                pass
    return True


def _normalize_audio_input(audio_in):
    """Safely extracts a valid file path or audio object from various Gradio component types."""
    if audio_in is None:
        return None
    if isinstance(audio_in, str):
        return audio_in.strip()
    if hasattr(audio_in, "name"):
        return audio_in.name
    if isinstance(audio_in, dict):
        return audio_in.get("name") or audio_in.get("path")
    return audio_in


def extract_voice_prompt_safely(model: OmniVoice, audio_path: str, ref_txt: str = None, progress_cb=None):
    """
    Creates a VoiceClonePrompt safely. If ref_txt is empty and ASR is unavailable/fails,
    falls back gracefully with a neutral reference text to avoid crash.
    Also handles silence trimming retry if audio is quiet.
    """
    audio_path = _normalize_audio_input(audio_path)
    if not audio_path:
        raise ValueError("Đường dẫn file âm thanh mẫu không hợp lệ.")

    if progress_cb:
        progress_cb(0.3, desc="Đang phân tích âm thanh mẫu...")
        
    actual_ref_text = (ref_txt or "").strip()
    
    # Check if we need ASR and whether ASR is ready
    if not actual_ref_text:
        try:
            if hasattr(model, "_asr_pipe") and model._asr_pipe is not None:
                if progress_cb:
                    progress_cb(0.5, desc="Đang nhận diện giọng nói qua Whisper ASR...")
                actual_ref_text = model.transcribe(audio_path)
            elif hasattr(model, "load_asr_model"):
                try:
                    if progress_cb:
                        progress_cb(0.4, desc="Đang kích hoạt Whisper ASR...")
                    model.load_asr_model()
                    actual_ref_text = model.transcribe(audio_path)
                except Exception as asr_err:
                    logging.warning(f"ASR load failed ({asr_err}), falling back to default heuristic text.")
                    actual_ref_text = "Giọng đọc mẫu tham chiếu chuẩn."
        except Exception as e:
            logging.warning(f"Transcription failed ({e}), using fallback reference text.")
            actual_ref_text = "Giọng đọc mẫu tham chiếu chuẩn."

    if not actual_ref_text:
        actual_ref_text = "Giọng đọc mẫu tham chiếu chuẩn."

    if progress_cb:
        progress_cb(0.7, desc="Đang trích xuất vector đặc trưng (Neural Audio Codec)...")
        
    try:
        prompt_obj = model.create_voice_clone_prompt(
            ref_audio=audio_path,
            ref_text=actual_ref_text,
            preprocess_prompt=True,
        )
    except Exception as e:
        logging.warning(f"Prompt creation with preprocess_prompt=True failed ({e}), retrying without silence trimming...")
        prompt_obj = model.create_voice_clone_prompt(
            ref_audio=audio_path,
            ref_text=actual_ref_text,
            preprocess_prompt=False,
        )
    return prompt_obj, actual_ref_text




# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


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


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:
    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
        voice_clone_prompt=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if voice_clone_prompt is not None:
                kw["voice_clone_prompt"] = voice_clone_prompt
            else:
                if not ref_audio:
                    return None, "Please upload a reference audio."
                kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    # =====================================================================
    # UI Design System (Monochrome Minimalist: Pure White / Neutral Gray / Deep Dark)
    # =====================================================================
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        font=["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        block_background_fill="white",
        block_background_fill_dark="*neutral_900",
        block_border_width="1px",
        block_border_color="*neutral_200",
        block_border_color_dark="*neutral_800",
        block_radius="10px",
        button_primary_background_fill="*neutral_900",
        button_primary_background_fill_dark="*neutral_100",
        button_primary_text_color="white",
        button_primary_text_color_dark="*neutral_900",
        button_primary_border_color="*neutral_900",
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_dark="*neutral_800",
        button_secondary_text_color="*neutral_900",
        button_secondary_text_color_dark="*neutral_100",
        button_secondary_border_color="*neutral_300",
        input_background_fill="*neutral_50",
        input_background_fill_dark="*neutral_950",
        input_border_color="*neutral_300",
        input_border_color_dark="*neutral_700",
        input_radius="8px",
    )

    css = """
    :root {
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .gradio-container {
        max-width: 1440px !important;
        margin: 0 auto !important;
        font-family: var(--font-sans) !important;
    }
    
    /* Header hero styling */
    .app-header {
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .dark .app-header {
        background: linear-gradient(180deg, #171717 0%, #0a0a0a 100%);
        border: 1px solid #262626;
    }
    .app-title {
        font-size: 24px !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        margin: 0 0 4px 0 !important;
        color: #111111 !important;
    }
    .dark .app-title {
        color: #ffffff !important;
    }
    .app-subtitle {
        font-size: 14px !important;
        color: #737373 !important;
        margin: 0 !important;
    }

    /* Tabs refinement */
    .tabs {
        border-bottom: 1px solid #e5e5e5 !important;
    }
    .tab-nav button {
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: -0.01em !important;
        padding: 10px 18px !important;
        border-radius: 8px 8px 0 0 !important;
    }
    .tab-nav button.selected {
        border-bottom: 2px solid #171717 !important;
        color: #171717 !important;
    }
    .dark .tab-nav button.selected {
        border-bottom: 2px solid #ffffff !important;
        color: #ffffff !important;
    }

    /* Buttons */
    button {
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        transition: all 0.15s ease-in-out !important;
    }
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
    }
    button:active {
        transform: translateY(0);
    }
    
    /* Preset Chips */
    .preset-chip {
        font-size: 13px !important;
        font-weight: 500 !important;
        border-radius: 20px !important;
        padding: 4px 14px !important;
        background: #f5f5f5 !important;
        border: 1px solid #e5e5e5 !important;
        color: #262626 !important;
    }
    .preset-chip:hover {
        background: #171717 !important;
        color: #ffffff !important;
        border-color: #171717 !important;
    }
    .dark .preset-chip {
        background: #262626 !important;
        border-color: #404040 !important;
        color: #e5e5e5 !important;
    }
    .dark .preset-chip:hover {
        background: #ffffff !important;
        color: #171717 !important;
        border-color: #ffffff !important;
    }

    /* Audio components */
    .compact-audio audio {
        height: 48px !important;
    }
    
    /* Segment Audio Card Box */
    .segment-card {
        border: 1px solid #e5e5e5;
        border-radius: 10px;
        padding: 12px;
        background: #ffffff;
        margin-bottom: 8px;
    }
    .dark .segment-card {
        border: 1px solid #262626;
        background: #171717;
    }

    /* Clean subtle scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-thumb {
        background: #d4d4d4;
        border-radius: 4px;
    }
    .dark ::-webkit-scrollbar-thumb {
        background: #404040;
    }
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Ngôn ngữ (Language)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Mặc định 'Auto' để tự động nhận diện ngôn ngữ.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("⚙️ Cài đặt tạo giọng nâng cao (Generation Settings)", open=False):
            with gr.Row():
                turbo_draft = gr.Checkbox(
                    label="⚡ Chế độ nháp siêu tốc (Turbo Draft - 16 bước)",
                    value=False,
                    info="Bật để nghe thử nhanh nhịp điệu/cảm xúc kịch bản với tốc độ nhanh nhất (16 steps).",
                )
            with gr.Row():
                sp = gr.Slider(
                    0.5,
                    1.5,
                    value=1.0,
                    step=0.05,
                    label="Tốc độ nói (Speed)",
                    info="1.0 = chuẩn. >1 nói nhanh hơn, <1 nói chậm hơn.",
                    scale=1
                )
                du = gr.Number(
                    value=None,
                    label="Thời lượng cố định (giây)",
                    info="Để trống để dùng tốc độ. Điền số để ép đúng số giây mong muốn.",
                    scale=1
                )
            with gr.Row():
                ns = gr.Slider(
                    4,
                    64,
                    value=24,
                    step=1,
                    label="Số bước khử nhiễu (Inference Steps)",
                    info="Mặc định: 24 (Tối ưu tốc độ & chất lượng trên GPU Colab/T4).",
                    scale=1
                )
                gs = gr.Slider(
                    0.0,
                    4.0,
                    value=2.0,
                    step=0.1,
                    label="Độ bám sát chỉ dẫn (Guidance Scale / CFG)",
                    info="Mặc định: 2.0.",
                    scale=1
                )
            with gr.Row():
                dn = gr.Checkbox(
                    label="Khử nhiễu nền (Denoise)",
                    value=True,
                    info="Bật để lọc bớt tạp âm nền.",
                )
                pp = gr.Checkbox(
                    label="Tiền xử lý mẫu giọng (Preprocess Prompt)",
                    value=True,
                    info="Tự động cắt khoảng lặng ở giọng mẫu.",
                )
                po = gr.Checkbox(
                    label="Hậu xử lý kết quả (Postprocess Output)",
                    value=True,
                    info="Xóa bỏ khoảng lặng thừa ở cuối file audio sinh ra.",
                )

            turbo_draft.change(
                lambda is_turbo: 16 if is_turbo else 24,
                inputs=[turbo_draft],
                outputs=[ns]
            )

        return ns, gs, dn, sp, du, pp, po

    storage_badge = "☁️ <b>Lưu trữ: Google Drive</b> <i>(/content/drive/MyDrive/OmniVoice_Studio)</i>" if _IS_GDRIVE else "💻 <b>Lưu trữ: Cục bộ (Local)</b>"

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Studio") as demo:
        with gr.Row(elem_classes="app-header"):
            with gr.Column():
                gr.Markdown(
                    f"""
<h1 class="app-title">⚡ OmniVoice AI Studio</h1>
<p class="app-subtitle">Nền tảng lồng tiếng & Clone Voice kịch bản chuyên nghiệp cho hơn 600+ ngôn ngữ &nbsp;|&nbsp; {storage_badge}</p>
"""
                )

        with gr.Tabs():
            # ==============================================================
            # Voice Manager / Quản lý Hồ sơ Giọng Mẫu
            # ==============================================================
            with gr.TabItem("🎙️ Voice Manager / Quản lý Hồ sơ Giọng"):
                gr.Markdown(
                    """
### 🎙️ Quản lý & Lưu Trữ Hồ Sơ Giọng Mẫu Cố Định (.pt Embedding)
*Tại đây bạn có thể trích xuất vector giọng nói từ một đoạn âm thanh mẫu (3-10 giây) và lưu lại vĩnh viễn.*
*Các tab **Script Clone**, **Voice Clone** và **Batch Voice Clone** chỉ cần chọn tên giọng để sử dụng ngay mà **không cần upload hay mã hóa lại audio**.*
"""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vm_name = gr.Textbox(
                            label="Tên nhân vật / Hồ sơ giọng",
                            placeholder="Ví dụ: Lucan_Energetic, Nam_Ke_Chuyen, Nu_Truyen_Cam...",
                        )
                        vm_audio = gr.Audio(
                            label="File âm thanh mẫu (3–10 giây)",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        vm_text = gr.Textbox(
                            label="Văn bản giọng mẫu (Tùy chọn)",
                            placeholder="Để trống nếu muốn tự động nhận diện (ASR) hoặc dùng tham chiếu mặc định...",
                        )
                        vm_save_btn = gr.Button("⚡ Trích xuất & Lưu Hồ Sơ Giọng (.pt)", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📂 Danh sách Hồ sơ giọng đã lưu")
                        init_profiles = list_voice_profiles()
                        init_selected = init_profiles[0] if init_profiles else None
                        init_preview = get_voice_profile_preview(init_selected) if init_selected else None
                        
                        def _format_voice_info(name):
                            if not name:
                                return "*(Chưa có hồ sơ giọng nào được chọn)*"
                            meta = get_voice_profile_metadata(name)
                            disp = meta.get("display_name", name)
                            saved = meta.get("saved_at", "Không rõ")
                            ref_t = meta.get("ref_text", "*(Không có văn bản tham chiếu)*")
                            prev_p = get_voice_profile_preview(name)
                            prev_st = "🔊 Có file nghe thử" if prev_p else "🔇 Không có file nghe thử"
                            return f"""**Chi tiết hồ sơ giọng:**
- **Tên hiển thị:** `{disp}` (Tệp lưu: `{name}.pt`)
- **Thời gian tạo:** {saved} | **File nghe thử:** {prev_st}
- **Văn bản tham chiếu:** *"{ref_t}"*
"""

                        vm_profiles_list = gr.Dropdown(
                            label="Chọn hồ sơ giọng",
                            choices=init_profiles,
                            value=init_selected,
                            interactive=True,
                        )
                        with gr.Row():
                            vm_refresh_btn = gr.Button("🔄 Làm mới danh sách", size="sm")
                            vm_delete_btn = gr.Button("🗑️ Xóa hồ sơ này", size="sm", variant="stop")
                        
                        vm_preview_audio = gr.Audio(
                            label="Nghe thử giọng mẫu gốc đã lưu",
                            value=init_preview,
                            interactive=False,
                            elem_classes="compact-audio"
                        )
                        vm_info_md = gr.Markdown(value=_format_voice_info(init_selected))
                        
                        with gr.Accordion("🔊 Thử nghiệm đọc văn bản nhanh với giọng này", open=False):
                            with gr.Row():
                                vm_test_text = gr.Textbox(
                                    label="Văn bản test giọng",
                                    value="Xin chào, đây là giọng đọc thử nghiệm được tạo từ hồ sơ đã lưu.",
                                    lines=2,
                                    scale=3
                                )
                                vm_test_lang = _lang_dropdown("Ngôn ngữ", "Auto")
                            vm_test_btn = gr.Button("▶ Sinh giọng đọc thử", variant="secondary")
                            vm_test_audio = gr.Audio(label="Kết quả đọc thử", type="numpy")

                        vm_status = gr.Textbox(label="Trạng thái", lines=3)

                def _sync_all_tabs(target_profile=None, msg=""):
                    profiles = list_voice_profiles()
                    has_profiles = len(profiles) > 0
                    selected = target_profile if (target_profile and target_profile in profiles) else (profiles[0] if has_profiles else None)
                    
                    preview_aud = get_voice_profile_preview(selected) if selected else None
                    info_text = _format_voice_info(selected)
                    
                    src_val = "🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if has_profiles else "📤 Tải lên Audio mẫu mới"
                    
                    preset_grp_vis = gr.update(visible=has_profiles)
                    custom_grp_vis = gr.update(visible=not has_profiles)
                    src_radio_upd = gr.update(value=src_val)
                    dropdown_upd = gr.update(choices=profiles, value=selected)
                    preview_upd = gr.update(value=preview_aud)
                    
                    return (
                        # Voice Manager (4):
                        dropdown_upd, preview_upd, info_text, msg,
                        # Voice Clone (5):
                        src_radio_upd, preset_grp_vis, custom_grp_vis, dropdown_upd, preview_upd,
                        # Batch Voice Clone (5):
                        src_radio_upd, preset_grp_vis, custom_grp_vis, dropdown_upd, preview_upd,
                        # Script Clone (5):
                        src_radio_upd, preset_grp_vis, custom_grp_vis, dropdown_upd, preview_upd
                    )

                def _on_vm_save(name, audio_path, ref_txt, progress=gr.Progress()):
                    if not name or not str(name).strip():
                        return _sync_all_tabs(None, "❌ Lỗi: Vui lòng đặt tên cho hồ sơ giọng.")
                    if not audio_path:
                        return _sync_all_tabs(None, "❌ Lỗi: Vui lòng tải lên file âm thanh mẫu.")
                    
                    try:
                        prompt_obj, actual_ref_text = extract_voice_prompt_safely(
                            model=model,
                            audio_path=audio_path,
                            ref_txt=ref_txt,
                            progress_cb=progress
                        )
                        saved_name = save_voice_profile(
                            name=str(name).strip(),
                            prompt_obj=prompt_obj,
                            metadata={
                                "ref_text": actual_ref_text,
                                "created_at": str(os.path.getmtime(audio_path)) if (isinstance(audio_path, str) and os.path.exists(audio_path)) else "",
                            },
                            preview_audio_path=audio_path,
                        )
                        progress(1.0, desc="Đã lưu hồ sơ giọng thành công!")
                        return _sync_all_tabs(saved_name, f"✅ Đã lưu thành công hồ sơ giọng: '{saved_name}.pt'!")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        return _sync_all_tabs(None, f"❌ Lỗi khi trích xuất vector giọng: {e}")

                def _on_vm_select(profile_nm):
                    if not profile_nm:
                        return None, "*(Chưa có hồ sơ giọng nào được chọn)*", ""
                    preview_aud = get_voice_profile_preview(profile_nm)
                    info_text = _format_voice_info(profile_nm)
                    return preview_aud, info_text, f"Đã chọn hồ sơ giọng: {profile_nm}"

                def _on_vm_refresh():
                    return _sync_all_tabs(None, "Đã làm mới danh sách hồ sơ giọng trên toàn bộ hệ thống.")

                def _on_vm_delete(profile_nm):
                    if not profile_nm:
                        return _sync_all_tabs(None, "Chưa chọn hồ sơ cần xóa.")
                    delete_voice_profile(profile_nm)
                    return _sync_all_tabs(None, f"Đã xóa hồ sơ: {profile_nm}")

                def _on_vm_quick_test(profile_nm, test_text, lang):
                    if not profile_nm:
                        return None, "Vui lòng chọn hồ sơ giọng trước khi đọc thử."
                    if not test_text or not test_text.strip():
                        return None, "Vui lòng nhập văn bản đọc thử."
                    loaded_prompt, _ = load_voice_profile(profile_nm)
                    if loaded_prompt is None:
                        return None, f"Lỗi: Không tìm thấy file hồ sơ {profile_nm}.pt"
                    return _gen(
                        test_text.strip(),
                        lang,
                        None,
                        None,
                        32,
                        2.0,
                        True,
                        1.0,
                        None,
                        True,
                        True,
                        mode="clone",
                        ref_text=None,
                        voice_clone_prompt=loaded_prompt
                    )

                vm_test_btn.click(
                    _on_vm_quick_test,
                    inputs=[vm_profiles_list, vm_test_text, vm_test_lang],
                    outputs=[vm_test_audio, vm_status]
                )


            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )

                        vc_source_type = gr.Radio(
                            choices=["🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)", "📤 Tải lên Audio mẫu mới"],
                            value="🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if list_voice_profiles() else "📤 Tải lên Audio mẫu mới",
                            label="Nguồn Giọng Mẫu (Voice Source)"
                        )

                        with gr.Group(visible=bool(list_voice_profiles())) as vc_preset_group:
                            with gr.Row():
                                vc_saved_profile = gr.Dropdown(
                                    label="Chọn Hồ sơ giọng cố định",
                                    choices=list_voice_profiles(),
                                    value=list_voice_profiles()[0] if list_voice_profiles() else None,
                                    scale=3
                                )
                                vc_preset_preview = gr.Audio(
                                    label="Nghe thử",
                                    value=get_voice_profile_preview(list_voice_profiles()[0]) if list_voice_profiles() else None,
                                    interactive=False,
                                    scale=2,
                                    elem_classes="compact-audio"
                                )

                        with gr.Group(visible=not bool(list_voice_profiles())) as vc_custom_group:
                            vc_ref_audio = gr.Audio(
                                label="Reference Audio / 参考音频",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            gr.Markdown(
                                "<span style='font-size:0.85em;color:#888;'>"
                                "Recommended: 3–10 seconds audio. "
                                "</span>"
                            )
                            vc_ref_text = gr.Textbox(
                                label=("Reference Text (optional) / 参考音频文本（可选）"),
                                lines=2,
                                placeholder="Transcript of the reference audio. Leave empty to auto-transcribe via ASR models.",
                            )

                        def _on_vc_source_change(mode_choice):
                            is_preset = "Hồ sơ giọng có sẵn" in mode_choice
                            return gr.update(visible=is_preset), gr.update(visible=not is_preset)

                        vc_source_type.change(
                            _on_vc_source_change,
                            inputs=[vc_source_type],
                            outputs=[vc_preset_group, vc_custom_group]
                        )
                        vc_saved_profile.change(
                            lambda p: get_voice_profile_preview(p),
                            inputs=[vc_saved_profile],
                            outputs=[vc_preset_preview]
                        )

                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(
                    text, lang, source_type, saved_prof, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
                ):
                    loaded_prompt = None
                    actual_ref_audio = ref_aud
                    if "Hồ sơ giọng có sẵn" in source_type:
                        if not saved_prof:
                            return None, "❌ Lỗi: Vui lòng chọn một hồ sơ giọng đã lưu từ danh sách."
                        loaded_prompt, _ = load_voice_profile(saved_prof)
                        actual_ref_audio = None
                        if loaded_prompt is None:
                            return None, f"❌ Lỗi: Không tìm thấy file hồ sơ {saved_prof}.pt"

                    return _gen(
                        text,
                        lang,
                        actual_ref_audio,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                        voice_clone_prompt=loaded_prompt,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_source_type,
                        vc_saved_profile,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

            # ==============================================================
            # Batch Voice Clone
            # ==============================================================
            with gr.TabItem("Batch Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        bvc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        
                        bvc_source_type = gr.Radio(
                            choices=["🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)", "📤 Tải lên Audio mẫu mới"],
                            value="🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if list_voice_profiles() else "📤 Tải lên Audio mẫu mới",
                            label="Nguồn Giọng Mẫu (Voice Source)"
                        )

                        with gr.Group(visible=bool(list_voice_profiles())) as bvc_preset_group:
                            with gr.Row():
                                bvc_saved_profile = gr.Dropdown(
                                    label="Chọn Hồ sơ giọng cố định",
                                    choices=list_voice_profiles(),
                                    value=list_voice_profiles()[0] if list_voice_profiles() else None,
                                    scale=3
                                )
                                bvc_preset_preview = gr.Audio(
                                    label="Nghe thử",
                                    value=get_voice_profile_preview(list_voice_profiles()[0]) if list_voice_profiles() else None,
                                    interactive=False,
                                    scale=2,
                                    elem_classes="compact-audio"
                                )

                        with gr.Group(visible=not bool(list_voice_profiles())) as bvc_custom_group:
                            bvc_ref_audio = gr.Audio(
                                label="Shared Reference Audio / 共享参考音频",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            bvc_ref_text = gr.Textbox(
                                label="Shared Reference Text (optional) / 共享参考音频文本（可选）",
                                placeholder="Leave empty to auto-transcribe.",
                            )

                        def _on_bvc_source_change(mode_choice):
                            is_preset = "Hồ sơ giọng có sẵn" in mode_choice
                            return gr.update(visible=is_preset), gr.update(visible=not is_preset)

                        bvc_source_type.change(
                            _on_bvc_source_change,
                            inputs=[bvc_source_type],
                            outputs=[bvc_preset_group, bvc_custom_group]
                        )
                        bvc_saved_profile.change(
                            lambda p: get_voice_profile_preview(p),
                            inputs=[bvc_saved_profile],
                            outputs=[bvc_preset_preview]
                        )

                        with gr.Accordion("Shared Instruct (optional)", open=False):
                            bvc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            bvc_ns,
                            bvc_gs,
                            bvc_dn,
                            bvc_sp,
                            bvc_du,
                            bvc_pp,
                            bvc_po,
                        ) = _gen_settings()

                        bvc_text1 = gr.Textbox(label="Text 1 / 文本 1", lines=2)
                        bvc_text2 = gr.Textbox(label="Text 2 / 文本 2", lines=2)
                        bvc_text3 = gr.Textbox(label="Text 3 / 文本 3", lines=2)
                        bvc_text4 = gr.Textbox(label="Text 4 / 文本 4", lines=2)
                        bvc_text5 = gr.Textbox(label="Text 5 / 文本 5", lines=2)
                        bvc_btn = gr.Button("Batch Generate / 批量生成", variant="primary")
                    with gr.Column(scale=1):
                        bvc_audio1 = gr.Audio(
                            label="Output Voice 1 / 合成结果 1",
                            type="numpy",
                        )
                        bvc_audio2 = gr.Audio(
                            label="Output Voice 2 / 合成结果 2",
                            type="numpy",
                        )
                        bvc_audio3 = gr.Audio(
                            label="Output Voice 3 / 合成结果 3",
                            type="numpy",
                        )
                        bvc_audio4 = gr.Audio(
                            label="Output Voice 4 / 合成结果 4",
                            type="numpy",
                        )
                        bvc_audio5 = gr.Audio(
                            label="Output Voice 5 / 合成结果 5",
                            type="numpy",
                        )
                        bvc_status = gr.Textbox(label="Status / 状态", lines=5)

                def _batch_clone_fn(
                    lang, 
                    source_type,
                    saved_prof,
                    ref_audio,
                    ref_text,
                    text1,
                    text2,
                    text3,
                    text4,
                    text5,
                    instruct, ns, gs, dn, sp, du, pp, po
                ):
                    results = []
                    statuses = []
                    
                    prompt = None
                    actual_ref_audio = ref_audio
                    if "Hồ sơ giọng có sẵn" in source_type:
                        if not saved_prof:
                            return None, None, None, None, None, "❌ Lỗi: Vui lòng chọn một hồ sơ giọng đã lưu từ danh sách."
                        prompt, _ = load_voice_profile(saved_prof)
                        actual_ref_audio = None
                        if prompt is None:
                            return None, None, None, None, None, f"❌ Lỗi: Không tìm thấy hồ sơ {saved_prof}.pt"
                    elif ref_audio and str(ref_audio).strip():
                        try:
                            prompt = model.create_voice_clone_prompt(
                                ref_audio=ref_audio,
                                ref_text=ref_text or None,
                            )
                        except Exception as e:
                            return None, None, None, None, None, f"❌ Lỗi trích xuất audio mẫu: {e}"
                    else:
                        return None, None, None, None, None, "❌ Lỗi: Vui lòng chọn hồ sơ giọng có sẵn hoặc tải lên file âm thanh mẫu."
                    
                    texts = [text1, text2, text3, text4, text5]
                    
                    for i, t in enumerate(texts, 1):
                        if t and t.strip():
                            try:
                                res, stat = _gen(
                                    t.strip(),
                                    lang,
                                    actual_ref_audio,
                                    instruct,
                                    ns,
                                    gs,
                                    dn,
                                    sp,
                                    du,
                                    pp,
                                    po,
                                    mode="clone",
                                    ref_text=ref_text or None,
                                    voice_clone_prompt=prompt,
                                )
                                results.append(res)
                                statuses.append(f"Voice {i}: {stat}")
                            except Exception as e:
                                results.append(None)
                                statuses.append(f"Voice {i}: Error: {e}")
                        else:
                            results.append(None)
                            statuses.append(f"Voice {i}: Skipped (empty text)")
                            
                    return results[0], results[1], results[2], results[3], results[4], "\n".join(statuses)

                bvc_btn.click(
                    _batch_clone_fn,
                    inputs=[
                        bvc_lang,
                        bvc_source_type,
                        bvc_saved_profile,
                        bvc_ref_audio,
                        bvc_ref_text,
                        bvc_text1,
                        bvc_text2,
                        bvc_text3,
                        bvc_text4,
                        bvc_text5,
                        bvc_instruct,
                        bvc_ns,
                        bvc_gs,
                        bvc_dn,
                        bvc_sp,
                        bvc_du,
                        bvc_pp,
                        bvc_po,
                    ],
                    outputs=[bvc_audio1, bvc_audio2, bvc_audio3, bvc_audio4, bvc_audio5, bvc_status],
                )

            # ==============================================================
            # Script Clone
            # ==============================================================
            with gr.TabItem("Script Clone / Sinh giọng theo Kịch bản"):
                sc_page_state = gr.State(value=0)
                sc_cache_state = gr.State(value={})
                sc_temp_dir_state = gr.State(value="")

                with gr.Row():
                    with gr.Column(scale=1):
                        sc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        
                        gr.Markdown("### 1. Chọn Giọng Mẫu (Voice Profile / Custom Audio)")
                        sc_source_type = gr.Radio(
                            choices=["🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)", "📤 Tải lên Audio mẫu mới"],
                            value="🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if list_voice_profiles() else "📤 Tải lên Audio mẫu mới",
                            label="Nguồn Giọng Mẫu (Voice Source)"
                        )

                        with gr.Group(visible=bool(list_voice_profiles())) as sc_preset_group:
                            with gr.Row():
                                sc_saved_profile = gr.Dropdown(
                                    label="Chọn Hồ sơ giọng cố định",
                                    choices=list_voice_profiles(),
                                    value=list_voice_profiles()[0] if list_voice_profiles() else None,
                                    scale=3
                                )
                                sc_preset_preview = gr.Audio(
                                    label="Nghe thử",
                                    value=get_voice_profile_preview(list_voice_profiles()[0]) if list_voice_profiles() else None,
                                    interactive=False,
                                    scale=2,
                                    elem_classes="compact-audio"
                                )

                        with gr.Group(visible=not bool(list_voice_profiles())) as sc_custom_group:
                            sc_ref_audio = gr.Audio(
                                label="Shared Reference Audio (Optional for Voice Cloning) / Giọng nói mẫu dùng chung (Tùy chọn)",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            sc_ref_text = gr.Textbox(
                                label="Shared Reference Text (optional) / Văn bản giọng mẫu (Tùy chọn)",
                                placeholder="Leave empty to auto-transcribe.",
                            )

                        def _on_sc_source_change(mode_choice):
                            is_preset = "Hồ sơ giọng có sẵn" in mode_choice
                            return gr.update(visible=is_preset), gr.update(visible=not is_preset)

                        sc_source_type.change(
                            _on_sc_source_change,
                            inputs=[sc_source_type],
                            outputs=[sc_preset_group, sc_custom_group]
                        )
                        sc_saved_profile.change(
                            lambda p: get_voice_profile_preview(p),
                            inputs=[sc_saved_profile],
                            outputs=[sc_preset_preview]
                        )
                        
                        gr.Markdown("### 2. Paste Script / Dán Kịch bản timeline")
                        sc_script = gr.Textbox(
                            label="Script / Kịch bản",
                            lines=12,
                            placeholder="[#1] THỜI GIAN: 0.0 -> 5.0\nVĂN BẢN (JP): 日本のコンビニ...\nCẢM XÚC: Energetic\nHƯỚNG DẪN AI: High energy intro\n------------------------------------------",
                            value="""After clawing your way out of your automated smart-home trap, an even bigger disaster strikes your pockets: Apple Pay and credit cards instantly turn into worthless plastic junk!
Picture yourself pulling into the Starbucks Drive-thru, ordering an iced oat milk caramel macchiato with extra cold foam for nine whole dollars.
You casually flick your wrist, tapping your shiny Apple Watch against the contactless payment terminal, waiting for that sleek, reassuring digital "beep."
Instead, the barista shakes his head apologetically as the screen blares a loud error buzz: "Nationwide network blackout, bro. Cash only today, exact change preferred!"
You frantically dump your entire backpack across the passenger seat: all you can find is one rusty 1998 quarter, two paperclips, and faded Target receipts from six months ago.
In a society where millions of people go an entire year without ever touching a crisp paper dollar bill, caffeine addicts everywhere plunge into sheer financial paralysis.
Downtown, outside the big Chase and Bank of America branches, massive queues wrap around four city blocks with desperate citizens praying in front of dead, black ATM screens.
Over at local grocery supermarkets, pure retail comedy unfolds as cloud-based barcode scanners and digital inventory systems freeze in unison.
Cashiers dust off vintage Casio desktop calculators, manually typing in the price of every cereal box while squinting at tiny yellow price stickers on shelf edges.
Shoppers stand in mile-long checkout lines holding baskets of fresh avocados, while managers weigh vegetables on antique mechanical balance scales with swinging needles!"""
                        )

                        with gr.Accordion("📂 Nhập kịch bản từ tệp tin (File Import)", open=False):
                            with gr.Row():
                                sc_import_raw = gr.File(label="Import Kịch bản Raw (mỗi dòng 1 câu .txt/.md)", file_types=[".txt", ".md"], scale=1)
                                sc_import_std = gr.File(label="Import Kịch bản Chuẩn Timeline (.txt/.md)", file_types=[".txt", ".md"], scale=1)

                        def _read_file_content(file_obj):
                            if not file_obj:
                                return gr.update()
                            try:
                                path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
                                for enc in ["utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"]:
                                    try:
                                        with open(path, "r", encoding=enc) as f:
                                            return f.read()
                                    except (UnicodeDecodeError, UnicodeError):
                                        continue
                                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                    return f.read()
                            except Exception as e:
                                gr.Warning(f"Lỗi đọc file: {e}")
                                return gr.update()

                        sc_import_raw.change(_read_file_content, inputs=[sc_import_raw], outputs=[sc_script])
                        sc_import_std.change(_read_file_content, inputs=[sc_import_std], outputs=[sc_script])

                        gr.Markdown("### 🤖 Tự động nhận diện cảm xúc kịch bản bằng Gemini AI")
                        with gr.Accordion("⚙️ Cấu hình Gemini Flash AI", open=False):
                            with gr.Row():
                                gemini_api_key = gr.Textbox(
                                    label="Gemini API Key",
                                    type="password",
                                    placeholder="Dán Google Gemini API Key vào đây (hoặc để trống nếu đã set ENV)...",
                                    value=os.environ.get("GEMINI_API_KEY", ""),
                                    scale=3
                                )
                                gemini_model = gr.Dropdown(
                                    label="Phiên bản mô hình",
                                    choices=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
                                    value="gemini-2.5-flash",
                                    scale=1
                                )
                        
                        gemini_analyze_btn = gr.Button("✨ Phân tích kịch bản & Gợi ý cảm xúc (Gemini AI)", variant="secondary")

                        with gr.Group(visible=False) as gemini_preview_group:
                            gr.Markdown("### 📋 Kết quả gợi ý từ Gemini AI (Xem trước & Chấp nhận)")
                            gemini_suggested_script = gr.Textbox(
                                label="Kịch bản sau khi Gemini gắn thẻ cảm xúc & hướng dẫn AI",
                                lines=8,
                                interactive=True,
                            )
                            with gr.Row():
                                gemini_apply_btn = gr.Button("✅ Đồng ý & Áp dụng vào kịch bản chính", variant="primary", scale=2)
                                gemini_cancel_btn = gr.Button("❌ Hủy bỏ", variant="secondary", scale=1)

                        with gr.Row():
                            sc_export_btn = gr.Button("💾 Xuất file Kịch bản chuẩn (.txt)", size="sm", scale=1)
                            sc_export_file = gr.File(label="Tải về Kịch bản chuẩn", interactive=False, scale=2)

                        def _on_export_script(script_text):
                            if not script_text or not script_text.strip():
                                gr.Warning("Kịch bản hiện tại đang trống.")
                                return None
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_standard_script.txt", mode="w", encoding="utf-8")
                            tmp.write(script_text)
                            tmp.close()
                            return tmp.name

                        sc_export_btn.click(_on_export_script, inputs=[sc_script], outputs=[sc_export_file])

                        gr.Markdown("💡 **Gợi ý cảm xúc nhanh:**")
                        with gr.Row():
                            preset_btn_fun = gr.Button("😂 Hài hước / Sôi nổi", size="sm", elem_classes="preset-chip")
                            preset_btn_serious = gr.Button("🧐 Nghiêm túc / Chỉnh chu", size="sm", elem_classes="preset-chip")
                            preset_btn_whisper = gr.Button("🤫 Thì thầm / Bí ẩn", size="sm", elem_classes="preset-chip")
                            preset_btn_dramatic = gr.Button("🔥 Kịch tính / Cao trào", size="sm", elem_classes="preset-chip")
                            preset_btn_calm = gr.Button("☕ Nhẹ nhàng / Bình thản", size="sm", elem_classes="preset-chip")

                        def _append_preset(script_text, emotion_str, guide_str):
                            preset_template = f"\nCẢM XÚC: {emotion_str}\nHƯỚNG DẪN AI: {guide_str}\n"
                            return (script_text or "") + preset_template

                        preset_btn_fun.click(lambda s: _append_preset(s, "Hài hước, vui vẻ", "High energy intro"), inputs=[sc_script], outputs=[sc_script])
                        preset_btn_serious.click(lambda s: _append_preset(s, "Nghiêm túc, chỉnh chu", "Steady pace, formal"), inputs=[sc_script], outputs=[sc_script])
                        preset_btn_whisper.click(lambda s: _append_preset(s, "Thì thầm", "Whisper, secret voice"), inputs=[sc_script], outputs=[sc_script])
                        preset_btn_dramatic.click(lambda s: _append_preset(s, "Kịch tính, cao trào", "High pitch, exciting"), inputs=[sc_script], outputs=[sc_script])
                        preset_btn_calm.click(lambda s: _append_preset(s, "Bình thường, tự nhiên", "Calm, steady pace"), inputs=[sc_script], outputs=[sc_script])

                        def _on_gemini_analyze(script_text, api_key, model_nm, progress=gr.Progress()):
                            if not script_text or not script_text.strip():
                                gr.Warning("Vui lòng dán kịch bản vào ô trước khi phân tích.")
                                return gr.update(visible=False), "", "Vui lòng nhập kịch bản trước."
                            progress(0.3, desc="Đang gửi kịch bản đến Gemini AI để phân tích ngữ cảnh...")
                            try:
                                suggested = analyze_script_with_gemini(script_text, api_key, model_nm)
                                progress(1.0, desc="Đã phân tích xong cảm xúc cho các phân đoạn!")
                                return gr.update(visible=True), suggested, "✅ Gemini đã phân tích xong! Vui lòng xem trước và bấm 'Đồng ý & Áp dụng'."
                            except Exception as e:
                                gr.Warning(f"Lỗi phân tích Gemini: {e}")
                                return gr.update(visible=False), "", f"Lỗi Gemini: {e}"

                        def _on_gemini_apply(suggested_text):
                            return suggested_text, gr.update(visible=False), "✅ Đã áp dụng kịch bản có gắn thẻ cảm xúc từ Gemini!"

                        def _on_gemini_cancel():
                            return gr.update(visible=False), "Đã hủy gợi ý của Gemini."
                        
                        (
                            sc_ns,
                            sc_gs,
                            sc_dn,
                            sc_sp,
                            sc_du,
                            sc_pp,
                            sc_po,
                        ) = _gen_settings()

                        with gr.Row():
                            sc_resume = gr.Checkbox(
                                label="🔄 Tiếp tục tiến trình (Bỏ qua các câu đã tạo thành công)",
                                value=True,
                                info="Tự động giữ nguyên các file âm thanh đã sinh trước đó, không tốn GPU sinh lại từ đầu.",
                            )
                        with gr.Row():
                            sc_btn = gr.Button("▶ Sinh đợt này (10 phân đoạn)", variant="primary", scale=2)
                            sc_next_btn = gr.Button("⏭ Đợt tiếp theo", variant="secondary", scale=1)
                        sc_all_btn = gr.Button("⚡ Sinh TOÀN BỘ kịch bản", variant="primary")
                    with gr.Column(scale=1):
                        with gr.Row():
                            sc_prev_view_btn = gr.Button("◀ Đợt trước", size="sm", scale=1)
                            sc_page_info = gr.Markdown("### 📑 Đang xem: Phân đoạn 1 - 10", elem_classes="text-center")
                            sc_next_view_btn = gr.Button("Đợt sau ▶", size="sm", scale=1)

                        sc_audios = []
                        sc_retries = []
                        with gr.Group():
                            for i in range(1, 11):
                                with gr.Row(elem_classes="segment-card"):
                                    aud = gr.Audio(label=f"Segment {i} Output", type="numpy", scale=4)
                                    btn = gr.Button("🔄 Thử lại", size="sm", scale=1)
                                    sc_audios.append(aud)
                                    sc_retries.append(btn)
                        
                        (
                            sc_audio1, sc_audio2, sc_audio3, sc_audio4, sc_audio5,
                            sc_audio6, sc_audio7, sc_audio8, sc_audio9, sc_audio10
                        ) = sc_audios
                        (
                            sc_retry1, sc_retry2, sc_retry3, sc_retry4, sc_retry5,
                            sc_retry6, sc_retry7, sc_retry8, sc_retry9, sc_retry10
                        ) = sc_retries
                        
                        sc_zip = gr.File(label="Download All WAVs (ZIP) / Tải xuống tất cả các tệp (ZIP)")
                        sc_parsed_markdown = gr.Markdown(label="Parsed Script Summary / Tóm tắt kịch bản đã phân tích")
                        sc_status = gr.Textbox(label="Status & Live Logs / Tiến trình trực tiếp", lines=5)

                PAGE_SIZE = 10

                def _generate_segments_core(
                    lang, source_type, saved_prof, ref_audio, ref_text, script_text,
                    ns, gs, dn, sp, du, pp, po, resume_existing,
                    target_indices, current_page, all_cache, temp_dir,
                    progress=gr.Progress()
                ):
                    if not script_text or not script_text.strip():
                        yield _render_script_page(0, [], {}, "", None, "Error: Script is empty.")
                        return
                    
                    try:
                        segments = parse_script(script_text)
                    except Exception as e:
                        yield _render_script_page(current_page, [], all_cache or {}, temp_dir or "", None, f"Error parsing script: {e}")
                        return
                        
                    if not segments:
                        yield _render_script_page(current_page, [], all_cache or {}, temp_dir or "", None, "Error: No valid segments found.")
                        return
                    
                    if all_cache is None:
                        all_cache = {}
                    
                    if not temp_dir or not os.path.exists(temp_dir):
                        if _IS_GDRIVE:
                            temp_dir = os.path.join(_OUTPUTS_DIR, "script_cache")
                            os.makedirs(temp_dir, exist_ok=True)
                        else:
                            temp_dir = tempfile.mkdtemp(prefix="omnivoice_script_")

                    prompt = None
                    actual_ref_audio = ref_audio
                    if "Hồ sơ giọng có sẵn" in source_type:
                        if not saved_prof:
                            yield _render_script_page(current_page, segments, all_cache, temp_dir, None, "❌ Lỗi: Vui lòng chọn một hồ sơ giọng đã lưu từ danh sách.")
                            return
                        prompt, _ = load_voice_profile(saved_prof)
                        actual_ref_audio = None
                        if prompt is None:
                            yield _render_script_page(current_page, segments, all_cache, temp_dir, None, f"❌ Lỗi: Không thể nạp hồ sơ giọng {saved_prof}.pt")
                            return
                    elif ref_audio and str(ref_audio).strip():
                        try:
                            prompt = model.create_voice_clone_prompt(
                                ref_audio=ref_audio,
                                ref_text=ref_text or None,
                            )
                        except Exception as e:
                            yield _render_script_page(current_page, segments, all_cache, temp_dir, None, f"❌ Lỗi trích xuất audio mẫu: {e}")
                            return
                    
                    mode = "clone" if (prompt is not None or (ref_audio and str(ref_audio).strip())) else "design"
                    statuses = []
                    total_targets = len(target_indices)

                    for step_i, idx in enumerate(target_indices):
                        if idx >= len(segments):
                            continue
                        seg = segments[idx]
                        seg_id = seg["id"]
                        text = seg["text"]
                        duration_val = seg["duration"]
                        instruct_val = seg["valid_instruct"]
                        wav_path = os.path.join(temp_dir, f"segment_{seg_id}.wav")

                        # Resume check: if audio already exists and is valid, load from disk
                        if resume_existing and os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                            if idx not in all_cache or all_cache[idx][0] is None:
                                try:
                                    import soundfile as sf
                                    cached_data, cached_sr = sf.read(wav_path, dtype="int16")
                                    all_cache[idx] = ((cached_sr, cached_data), "Đã có sẵn (Khôi phục)")
                                    statuses.append(f"Segment #{seg_id} [Đã khôi phục]: Bỏ qua để tiết kiệm GPU.")
                                except Exception:
                                    pass
                            if idx in all_cache and all_cache[idx][0] is not None:
                                continue

                        # Show current progress and exact text being generated
                        progress(
                            (step_i) / max(1, total_targets),
                            desc=f"[{step_i + 1}/{total_targets}] Đang sinh Segment #{seg_id}: {text[:35]}..."
                        )

                        try:
                            res, stat = _gen(
                                text,
                                lang,
                                actual_ref_audio if (actual_ref_audio and str(actual_ref_audio).strip()) else None,
                                instruct_val,
                                ns,
                                gs,
                                dn,
                                sp,
                                duration_val,
                                pp,
                                po,
                                mode,
                                ref_text or None,
                                prompt,
                            )
                            all_cache[idx] = (res, stat)
                            statuses.append(f"Segment #{seg_id} [Thành công]: {stat}")

                            if res and res[1] is not None:
                                import soundfile as sf
                                sf.write(wav_path, res[1], res[0])
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            all_cache[idx] = (None, f"Error: {e}")
                            statuses.append(f"Segment #{seg_id} [LỖI]: {e} (Các đoạn khác vẫn được bảo toàn)")

                        # Periodic GPU memory cleanup
                        if (step_i + 1) % 5 == 0:
                            _clean_gpu_memory()

                        # Update zip file with all generated wavs in temp_dir incrementally
                        zip_path = None
                        wav_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".wav")]
                        if wav_files:
                            zip_path = os.path.join(temp_dir, "all_segments.zip")
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for wp in wav_files:
                                    zipf.write(wp, os.path.basename(wp))

                        # Live yield update after EACH segment
                        yield _render_script_page(current_page, segments, all_cache, temp_dir, zip_path, "\n".join(statuses))

                    _clean_gpu_memory()
                    progress(1.0, desc="Hoàn tất sinh giọng!")
                    yield _render_script_page(current_page, segments, all_cache, temp_dir, zip_path, "\n".join(statuses))

                def _render_script_page(page_idx, segments, all_cache, temp_dir, zip_path, status_text=""):
                    N = len(segments)
                    P = max(1, (N + PAGE_SIZE - 1) // PAGE_SIZE)
                    page_idx = max(0, min(page_idx, P - 1))

                    start_idx = page_idx * PAGE_SIZE
                    end_idx = min(start_idx + PAGE_SIZE, N)

                    audio_updates = []
                    for slot in range(PAGE_SIZE):
                        actual_idx = start_idx + slot
                        if actual_idx < N:
                            seg = segments[actual_idx]
                            cached = all_cache.get(actual_idx, (None, ""))
                            audio_val = cached[0]
                            audio_label = f"Phân đoạn #{seg['id']} ({seg['duration']}s): {seg['text'][:25]}..."
                            audio_updates.append(gr.update(value=audio_val, label=audio_label, visible=True))
                        else:
                            audio_updates.append(gr.update(value=None, label="Trống", visible=False))

                    parsed_summary = "### Parsed Segments Summary:\n"
                    for idx, seg in enumerate(segments):
                        is_current = (start_idx <= idx < end_idx)
                        prefix = "👉 " if is_current else "- "
                        done_icon = " ✅" if idx in all_cache and all_cache[idx][0] is not None else (" ❌ (Lỗi)" if idx in all_cache else "")
                        inst_display = f"*{seg['raw_instruct']}*" if seg['raw_instruct'] else ""
                        mapped_display = f" [`{seg['valid_instruct']}`]" if seg['valid_instruct'] else ""
                        parsed_summary += f"{prefix}**Segment #{seg['id']}** ({seg['duration']}s): {inst_display}{mapped_display} - \"{seg['text'][:30]}...\"{done_icon}\n"

                    page_info_md = f"### 📑 Đang xem: Phân đoạn {start_idx + 1} - {end_idx} / Tổng: {N} (Trang {page_idx + 1}/{P})"

                    if not zip_path and temp_dir and os.path.exists(temp_dir):
                        z = os.path.join(temp_dir, "all_segments.zip")
                        if os.path.exists(z):
                            zip_path = z

                    return (
                        *audio_updates,
                        zip_path, parsed_summary, status_text, page_info_md,
                        page_idx, all_cache, temp_dir
                    )

                def _on_generate_current(lang, source_type, saved_prof, ref_audio, ref_text, script_text, ns, gs, dn, sp, du, pp, po, resume_existing, page_idx, all_cache, temp_dir, progress=gr.Progress()):
                    segments = parse_script(script_text) if script_text else []
                    if not segments:
                        yield _render_script_page(0, [], {}, "", None, "Error: Script is empty.")
                        return
                    start_idx = page_idx * PAGE_SIZE
                    target_indices = list(range(start_idx, min(start_idx + PAGE_SIZE, len(segments))))
                    for res in _generate_segments_core(lang, source_type, saved_prof, ref_audio, ref_text, script_text, ns, gs, dn, sp, du, pp, po, resume_existing, target_indices, page_idx, all_cache, temp_dir, progress):
                        yield res

                def _on_continue_next(lang, source_type, saved_prof, ref_audio, ref_text, script_text, ns, gs, dn, sp, du, pp, po, resume_existing, page_idx, all_cache, temp_dir, progress=gr.Progress()):
                    segments = parse_script(script_text) if script_text else []
                    if not segments:
                        yield _render_script_page(0, [], {}, "", None, "Error: Script is empty.")
                        return
                    N = len(segments)
                    P = max(1, (N + PAGE_SIZE - 1) // PAGE_SIZE)
                    next_page = min(page_idx + 1, P - 1)
                    start_idx = next_page * PAGE_SIZE
                    target_indices = list(range(start_idx, min(start_idx + PAGE_SIZE, N)))
                    for res in _generate_segments_core(lang, source_type, saved_prof, ref_audio, ref_text, script_text, ns, gs, dn, sp, du, pp, po, resume_existing, target_indices, next_page, all_cache, temp_dir, progress):
                        yield res

                def _on_generate_all(lang, source_type, saved_prof, ref_audio, ref_text, script_text, ns, gs, dn, sp, du, pp, po, resume_existing, page_idx, all_cache, temp_dir, progress=gr.Progress()):
                    segments = parse_script(script_text) if script_text else []
                    if not segments:
                        yield _render_script_page(0, [], {}, "", None, "Error: Script is empty.")
                        return
                    target_indices = list(range(len(segments)))
                    for res in _generate_segments_core(lang, source_type, saved_prof, ref_audio, ref_text, script_text, ns, gs, dn, sp, du, pp, po, resume_existing, target_indices, page_idx, all_cache, temp_dir, progress):
                        yield res

                def _on_retry_single(slot_idx, lang, source_type, saved_prof, ref_audio, ref_text, script_text, ns, gs, dn, sp, du, pp, po, resume_existing, page_idx, all_cache, temp_dir, progress=gr.Progress()):
                    segments = parse_script(script_text) if script_text else []
                    actual_idx = page_idx * PAGE_SIZE + slot_idx
                    if not segments or actual_idx >= len(segments):
                        yield _render_script_page(page_idx, segments, all_cache or {}, temp_dir or "", None, f"Phân đoạn {actual_idx + 1} không tồn tại.")
                        return
                    # Force resume_existing=False when explicitly retrying a specific segment
                    for res in _generate_segments_core(lang, source_type, saved_prof, ref_audio, ref_text, script_text, ns, gs, dn, sp, du, pp, po, False, [actual_idx], page_idx, all_cache, temp_dir, progress):
                        yield res

                def _on_prev_view(script_text, page_idx, all_cache, temp_dir):
                    segments = parse_script(script_text) if script_text else []
                    new_page = max(0, page_idx - 1)
                    return _render_script_page(new_page, segments, all_cache or {}, temp_dir, None, f"Đang xem trang {new_page + 1}")

                def _on_next_view(script_text, page_idx, all_cache, temp_dir):
                    segments = parse_script(script_text) if script_text else []
                    N = len(segments)
                    P = max(1, (N + PAGE_SIZE - 1) // PAGE_SIZE)
                    new_page = min(page_idx + 1, P - 1)
                    return _render_script_page(new_page, segments, all_cache or {}, temp_dir, None, f"Đang xem trang {new_page + 1}")

                gen_inputs = [
                    sc_lang, sc_source_type, sc_saved_profile, sc_ref_audio, sc_ref_text, sc_script,
                    sc_ns, sc_gs, sc_dn, sc_sp, sc_du, sc_pp, sc_po, sc_resume,
                    sc_page_state, sc_cache_state, sc_temp_dir_state
                ]
                gen_outputs = [
                    *sc_audios,
                    sc_zip, sc_parsed_markdown, sc_status, sc_page_info,
                    sc_page_state, sc_cache_state, sc_temp_dir_state
                ]

                sc_btn.click(_on_generate_current, inputs=gen_inputs, outputs=gen_outputs)
                sc_next_btn.click(_on_continue_next, inputs=gen_inputs, outputs=gen_outputs)
                sc_all_btn.click(_on_generate_all, inputs=gen_inputs, outputs=gen_outputs)

                for slot_i, btn in enumerate(sc_retries):
                    btn.click(lambda *args, s=slot_i: _on_retry_single(s, *args), inputs=gen_inputs, outputs=gen_outputs)

                sc_prev_view_btn.click(
                    _on_prev_view,
                    inputs=[sc_script, sc_page_state, sc_cache_state, sc_temp_dir_state],
                    outputs=gen_outputs
                )
                sc_next_view_btn.click(
                    _on_next_view,
                    inputs=[sc_script, sc_page_state, sc_cache_state, sc_temp_dir_state],
                    outputs=gen_outputs
                )

                gemini_analyze_btn.click(
                    _on_gemini_analyze,
                    inputs=[sc_script, gemini_api_key, gemini_model],
                    outputs=[gemini_preview_group, gemini_suggested_script, sc_status]
                )
                gemini_apply_btn.click(
                    _on_gemini_apply,
                    inputs=[gemini_suggested_script],
                    outputs=[sc_script, gemini_preview_group, sc_status]
                )
                gemini_cancel_btn.click(
                    _on_gemini_cancel,
                    outputs=[gemini_preview_group, sc_status]
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

            # ==============================================================
            # Audio Merger (Ghép Audio)
            # ==============================================================
            with gr.TabItem("🧩 Ghép Audio"):
                gr.Markdown(
                    """
### 🧩 Ghép Nhiều File Audio Thành 1 File Âm Thanh Hoàn Chỉnh
*Tự động phát hiện và sắp xếp thứ tự chính xác theo số trong tên file (ví dụ: `audio_1.wav` ➔ `audio_2.wav` ➔ ... ➔ `audio_10.wav`).*
*Hỗ trợ tải lên danh sách file hoặc quét trực tiếp thư mục (mặc định: `audio/no_internet`).*
"""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        am_mode = gr.Radio(
                            label="Nguồn file Audio",
                            choices=["Quét thư mục cục bộ (Local Folder)", "Upload file trực tiếp"],
                            value="Quét thư mục cục bộ (Local Folder)",
                        )
                        am_folder = gr.Textbox(
                            label="Đường dẫn thư mục chứa audio",
                            value="audio/no_internet",
                            placeholder="Ví dụ: audio/no_internet hoặc C:/path/to/audios",
                            visible=True,
                        )
                        am_upload_files = gr.File(
                            label="Tải lên danh sách file Audio",
                            file_count="multiple",
                            file_types=["audio"],
                            visible=False,
                        )
                        
                        am_gap = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=0.3,
                            step=0.1,
                            label="Khoảng lặng giữa các phân đoạn (giây)",
                            info="Chèn thêm khoảng im lặng ngắn giữa các audio để giọng đọc tự nhiên hơn.",
                        )
                        am_merge_btn = gr.Button("🚀 Bắt Đầu Ghép Audio", variant="primary")

                    with gr.Column(scale=1):
                        am_status = gr.Textbox(
                            label="Trạng thái & Thứ tự các file đã nhận diện",
                            lines=8,
                            placeholder="Thông tin thứ tự các file sau khi sắp xếp sẽ hiển thị tại đây...",
                        )
                        am_output_audio = gr.Audio(
                            label="🔊 Audio Đã Ghép Hoàn Chỉnh",
                            type="filepath",
                        )
                        am_download_audio = gr.File(
                            label="💾 Tải File Audio Về Máy (.wav)",
                            visible=True,
                        )

                def _toggle_am_mode(choice):
                    if choice == "Quét thư mục cục bộ (Local Folder)":
                        return gr.update(visible=True), gr.update(visible=False)
                    else:
                        return gr.update(visible=False), gr.update(visible=True)

                am_mode.change(
                    _toggle_am_mode,
                    inputs=[am_mode],
                    outputs=[am_folder, am_upload_files],
                )

                def _natural_sort_key(file_path):
                    name = os.path.basename(file_path)
                    # Phân tích cụm số tự nhiên, hỗ trợ tên có nhiều chữ số (audio_1, audio_2, audio_10...)
                    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

                def _find_ffmpeg():
                    w = shutil.which("ffmpeg")
                    if w:
                        return w
                    winget_path = r"C:\Users\ADMIN\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
                    if os.path.exists(winget_path):
                        return winget_path
                    return "ffmpeg"

                def _process_audio_merger(mode, folder_path, uploaded_files, gap_sec, progress=gr.Progress()):
                    input_paths = []
                    valid_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

                    if mode == "Quét thư mục cục bộ (Local Folder)":
                        clean_folder = (folder_path or "").strip().strip('"').strip("'")
                        if not clean_folder:
                            return "❌ Lỗi: Vui lòng nhập đường dẫn thư mục chứa audio.", None, None
                        if not os.path.exists(clean_folder):
                            return f"❌ Lỗi: Thư mục không tồn tại: {clean_folder}", None, None
                        if not os.path.isdir(clean_folder):
                            return f"❌ Lỗi: Đường dẫn không phải là thư mục: {clean_folder}", None, None

                        for f in os.listdir(clean_folder):
                            full_f = os.path.join(clean_folder, f)
                            if os.path.isfile(full_f) and f.lower().endswith(valid_exts) and os.path.getsize(full_f) > 0:
                                input_paths.append(full_f)
                    else:
                        if not uploaded_files:
                            return "❌ Lỗi: Vui lòng tải lên ít nhất một file audio.", None, None
                        for item in uploaded_files:
                            # Hỗ trợ cả Gradio 3, 4, 5 (str, dict, hoặc object có thuộc tính .name)
                            if isinstance(item, str):
                                p = item
                            elif hasattr(item, "name"):
                                p = item.name
                            elif isinstance(item, dict) and "name" in item:
                                p = item["name"]
                            else:
                                p = str(item)
                            
                            if p and os.path.exists(p) and os.path.getsize(p) > 0:
                                input_paths.append(p)

                    if not input_paths:
                        return "❌ Lỗi: Không tìm thấy file audio nào hợp lệ (hoặc các file đều rỗng 0 bytes).", None, None

                    # Sắp xếp tự nhiên theo số trong tên file
                    input_paths.sort(key=_natural_sort_key)

                    out_dir = tempfile.mkdtemp(prefix="omnivoice_merge_")
                    merged_wav_path = os.path.join(out_dir, "merged_audio.wav")
                    file_info_lines = ["📋 DANH SÁCH & THỨ TỰ GHÉP CHÍNH XÁC:"]

                    progress(0.15, desc="Đang phân tích các file âm thanh...")

                    # Cách 1: Thử ghép bằng soundfile + numpy
                    merged_successfully = False
                    try:
                        import soundfile as sf
                        audio_arrays = []
                        target_sr = None

                        for i, p in enumerate(input_paths):
                            fname = os.path.basename(p)
                            try:
                                data, sr = sf.read(p)
                                if data.ndim > 1:
                                    data = np.mean(data, axis=1) # Mono
                                
                                if target_sr is None:
                                    target_sr = sr
                                elif sr != target_sr:
                                    try:
                                        import librosa
                                        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
                                    except Exception:
                                        pass

                                dur = len(data) / target_sr
                                file_info_lines.append(f"  [{i + 1}] {fname} - Thời lượng: {dur:.2f}s")
                                audio_arrays.append(data)
                            except Exception as e:
                                file_info_lines.append(f"  [{i + 1}] {fname} - [Lỗi đọc]: {e}")

                        if len(audio_arrays) == len(input_paths):
                            gap_samples = int(max(0.0, float(gap_sec or 0)) * target_sr)
                            silence = np.zeros(gap_samples, dtype=np.float32) if gap_samples > 0 else np.array([], dtype=np.float32)

                            merged_list = []
                            for i, arr in enumerate(audio_arrays):
                                merged_list.append(arr)
                                if i < len(audio_arrays) - 1 and len(silence) > 0:
                                    merged_list.append(silence)

                            merged_audio = np.concatenate(merged_list)
                            total_duration = len(merged_audio) / target_sr
                            sf.write(merged_wav_path, merged_audio, target_sr)
                            merged_successfully = True
                    except Exception:
                        merged_successfully = False

                    # Cách 2: Ghép bằng FFmpeg (Concat Demuxer an toàn, không giới hạn số lượng file)
                    if not merged_successfully:
                        progress(0.4, desc="Đang ghép nối các file qua FFmpeg...")
                        ffmpeg_bin = _find_ffmpeg()
                        file_info_lines = ["📋 DANH SÁCH & THỨ TỰ GHÉP CHÍNH XÁC (FFmpeg):"]

                        # Chuẩn bị file danh sách concat
                        concat_list_file = os.path.join(out_dir, "concat_list.txt")
                        silence_wav_path = None
                        gap_val = max(0.0, float(gap_sec or 0))

                        # Tạo file silence nếu có gap
                        if gap_val > 0:
                            silence_wav_path = os.path.join(out_dir, "silence_gap.wav")
                            cmd_silence = [
                                ffmpeg_bin, "-y",
                                "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
                                "-t", f"{gap_val:.3f}",
                                silence_wav_path
                            ]
                            subprocess.run(cmd_silence, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        with open(concat_list_file, "w", encoding="utf-8") as f_txt:
                            for i, p in enumerate(input_paths):
                                fname = os.path.basename(p)
                                file_info_lines.append(f"  [{i + 1}] {fname}")
                                escaped_p = os.path.abspath(p).replace(os.sep, "/")
                                f_txt.write(f"file '{escaped_p}'\n")
                                if i < len(input_paths) - 1 and silence_wav_path and os.path.exists(silence_wav_path):
                                    escaped_silence = os.path.abspath(silence_wav_path).replace(os.sep, "/")
                                    f_txt.write(f"file '{escaped_silence}'\n")

                        cmd_concat = [
                            ffmpeg_bin, "-y",
                            "-f", "concat",
                            "-safe", "0",
                            "-i", concat_list_file,
                            "-c:a", "pcm_s16le",
                            merged_wav_path
                        ]
                        
                        try:
                            res = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
                            if res.returncode == 0 and os.path.exists(merged_wav_path) and os.path.getsize(merged_wav_path) > 0:
                                merged_successfully = True
                            else:
                                return f"❌ Lỗi khi ghép audio qua FFmpeg: {res.stderr[-400:]}", None, None
                        except Exception as e:
                            return f"❌ Lỗi chạy lệnh FFmpeg: {e}", None, None

                    file_info_lines.append("--------------------------------------------")
                    file_info_lines.append(f"✅ ĐÃ GHÉP THÀNH CÔNG {len(input_paths)} FILE AUDIO!")
                    file_info_lines.append(f"📁 File âm thanh hoàn chỉnh: {merged_wav_path}")

                    progress(1.0, desc="Hoàn tất ghép audio!")
                    return "\n".join(file_info_lines), merged_wav_path, merged_wav_path

                am_merge_btn.click(
                    _process_audio_merger,
                    inputs=[am_mode, am_folder, am_upload_files, am_gap],
                    outputs=[am_status, am_output_audio, am_download_audio],
                )

        # Cross-tab Event Listeners: Đăng ký liên kết đồng bộ hóa 100% giữa Voice Manager và tất cả các tab
        sync_all_outputs = [
            # Voice Manager (4):
            vm_profiles_list, vm_preview_audio, vm_info_md, vm_status,
            # Voice Clone (5):
            vc_source_type, vc_preset_group, vc_custom_group, vc_saved_profile, vc_preset_preview,
            # Batch Voice Clone (5):
            bvc_source_type, bvc_preset_group, bvc_custom_group, bvc_saved_profile, bvc_preset_preview,
            # Script Clone (5):
            sc_source_type, sc_preset_group, sc_custom_group, sc_saved_profile, sc_preset_preview
        ]

        vm_save_btn.click(
            _on_vm_save,
            inputs=[vm_name, vm_audio, vm_text],
            outputs=sync_all_outputs,
        )
        vm_profiles_list.change(
            _on_vm_select,
            inputs=[vm_profiles_list],
            outputs=[vm_preview_audio, vm_info_md, vm_status],
        )
        vm_refresh_btn.click(
            _on_vm_refresh,
            outputs=sync_all_outputs,
        )
        vm_delete_btn.click(
            _on_vm_delete,
            inputs=[vm_profiles_list],
            outputs=sync_all_outputs,
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    if torch.cuda.is_available() and "cuda" in str(device).lower():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("CUDA optimizations enabled: TF32=True, cuDNN benchmark=True")
        except Exception:
            pass

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=dtype,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
