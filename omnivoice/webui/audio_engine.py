import os
import re
import gc
import shutil
import tempfile
import subprocess
import logging
import numpy as np
import torch
import gradio as gr
from omnivoice import OmniVoice


def _clean_gpu_memory():
    """Safely frees cached CUDA VRAM and forces garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


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


def _natural_sort_key(file_path):
    """Sort strings containing numbers naturally (audio_1, audio_2, audio_10)."""
    name = os.path.basename(file_path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def _find_ffmpeg():
    """Finds ffmpeg executable path on system or Windows winget packages."""
    w = shutil.which("ffmpeg")
    if w:
        return w
    winget_path = r"C:\Users\ADMIN\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
    if os.path.exists(winget_path):
        return winget_path
    return "ffmpeg"


def process_audio_merger(mode, folder_path, uploaded_files, gap_sec, progress=gr.Progress()):
    """Merges multiple audio files into a single audio file with natural sorting and silence gaps."""
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

    input_paths.sort(key=_natural_sort_key)

    out_dir = tempfile.mkdtemp(prefix="omnivoice_merge_")
    merged_wav_path = os.path.join(out_dir, "merged_audio.wav")
    file_info_lines = ["📋 DANH SÁCH & THỨ TỰ GHÉP CHÍNH XÁC:"]

    progress(0.15, desc="Đang phân tích các file âm thanh...")

    # Method 1: Soundfile + Librosa
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
            sf.write(merged_wav_path, merged_audio, target_sr)
            merged_successfully = True
    except Exception:
        merged_successfully = False

    # Method 2: FFmpeg concat demuxer fallback
    if not merged_successfully:
        progress(0.4, desc="Đang ghép nối các file qua FFmpeg...")
        ffmpeg_bin = _find_ffmpeg()
        file_info_lines = ["📋 DANH SÁCH & THỨ TỰ GHÉP CHÍNH XÁC (FFmpeg):"]

        concat_list_file = os.path.join(out_dir, "concat_list.txt")
        silence_wav_path = None
        gap_val = max(0.0, float(gap_sec or 0))

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
