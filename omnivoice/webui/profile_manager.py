import os
import re
import time
import shutil
import logging
import torch
from omnivoice.webui.config import _SAVED_VOICES_DIR


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
