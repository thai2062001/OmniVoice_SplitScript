import os
import gradio as gr
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Storage Directory Detection (Auto-detect Google Drive vs Local)
# ---------------------------------------------------------------------------
def _get_storage_dirs():
    """
    Detects if running on Google Colab with Google Drive mounted.
    Returns (saved_voices_dir, outputs_dir, is_gdrive).
    Also configures HF_HOME and TORCH_HOME to permanently cache model weights on Drive.
    """
    gdrive_base = "/content/drive/MyDrive/OmniVoice_Studio"
    if os.path.exists("/content/drive/MyDrive"):
        voices_dir = os.path.join(gdrive_base, "saved_voices")
        out_dir = os.path.join(gdrive_base, "outputs")
        hf_cache_dir = os.path.join(gdrive_base, "hf_cache")
        torch_cache_dir = os.path.join(gdrive_base, "torch_cache")
        
        # Persist model weights on Google Drive so they don't need re-downloading on every Colab restart
        os.environ.setdefault("HF_HOME", hf_cache_dir)
        os.environ.setdefault("TORCH_HOME", torch_cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
        
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.makedirs(torch_cache_dir, exist_ok=True)
        is_gdrive = True
    else:
        # Fallback to local workspace
        local_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        voices_dir = os.path.join(local_base, "cli", "saved_voices")
        out_dir = os.path.join(local_base, "cli", "outputs")
        is_gdrive = False

    os.makedirs(voices_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    return voices_dir, out_dir, is_gdrive


_SAVED_VOICES_DIR, _OUTPUTS_DIR, _IS_GDRIVE = _get_storage_dirs()

# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Voice Design instruction templates (Vietnamese UI)
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "Giới tính (Gender)": ["Nam (Male)", "Nữ (Female)"],
    "Độ tuổi (Age)": [
        "Trẻ em (Child)",
        "Thiếu niên (Teenager)",
        "Thanh niên (Young Adult)",
        "Trung niên (Middle-aged)",
        "Lớn tuổi (Elderly)",
    ],
    "Âm điệu (Pitch)": [
        "Cực kỳ trầm (Very Low Pitch)",
        "Trầm / Điềm đạm (Low Pitch)",
        "Vừa phải / Tự nhiên (Moderate Pitch)",
        "Cao / Sôi nổi (High Pitch)",
        "Cực cao / Phấn khích (Very High Pitch)",
    ],
    "Phong cách (Style)": ["Thì thầm / Tâm sự (Whisper)"],
    "Chất giọng Quốc tế (Accent)": [
        "Giọng Mỹ (American Accent)",
        "Giọng Úc (Australian Accent)",
        "Giọng Anh (British Accent)",
        "Giọng Trung Quốc (Chinese Accent)",
        "Giọng Canada (Canadian Accent)",
        "Giọng Ấn Độ (Indian Accent)",
        "Giọng Hàn Quốc (Korean Accent)",
        "Giọng Bồ Đào Nha (Portuguese Accent)",
        "Giọng Nga (Russian Accent)",
        "Giọng Nhật Bản (Japanese Accent)",
    ],
    "Phương ngữ tiếng Trung (Chinese Dialect)": [
        "Hà Nam (Henan Dialect / 河南话)",
        "Thiểm Tây (Shaanxi Dialect / 陕西话)",
        "Tứ Xuyên (Sichuan Dialect / 四川话)",
        "Quý Châu (Guizhou Dialect / 贵州话)",
        "Vân Nam (Yunnan Dialect / 云南话)",
        "Quế Lâm (Guilin Dialect / 桂林话)",
        "Tế Nam (Jinan Dialect / 济南话)",
        "Thạch Gia Trang (Shijiazhuang Dialect / 石家庄话)",
        "Cam Túc (Gansu Dialect / 甘肃话)",
        "Ninh Hạ (Ningxia Dialect / 宁夏话)",
        "Thanh Đảo (Qingdao Dialect / 青岛话)",
        "Đông Bắc (Northeast Dialect / 东北话)",
    ],
}

_ATTR_INFO = {
    "Chất giọng Quốc tế (Accent)": "Có hiệu lực rõ nhất khi đọc tiếng Anh.",
    "Phương ngữ tiếng Trung (Chinese Dialect)": "Có hiệu lực khi đọc tiếng Trung.",
}


def get_theme_and_css():
    """Builds and returns the Gradio Theme and CSS design tokens."""
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
        --brand-primary: #18181b;
        --brand-accent: #2563eb;
        --surface-card: #ffffff;
        --border-subtle: #e4e4e7;
    }
    .gradio-container {
        max-width: 1440px !important;
        margin: 0 auto !important;
        font-family: var(--font-sans) !important;
        padding: 16px 24px !important;
    }
    
    /* Header hero styling */
    .app-header {
        background: linear-gradient(135deg, #09090b 0%, #18181b 50%, #27272a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        color: #ffffff;
    }
    .app-title {
        font-size: 26px !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        margin: 0 0 6px 0 !important;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .app-subtitle {
        font-size: 14px !important;
        color: #a1a1aa !important;
        margin: 0 !important;
        line-height: 1.5;
    }

    /* Tabs refinement */
    .tabs {
        border-bottom: 2px solid #e4e4e7 !important;
        margin-bottom: 20px !important;
    }
    .dark .tabs {
        border-bottom: 2px solid #27272a !important;
    }
    .tab-nav button {
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: -0.01em !important;
        padding: 12px 20px !important;
        border-radius: 10px 10px 0 0 !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    .tab-nav button:hover {
        background: rgba(0, 0, 0, 0.04) !important;
    }
    .dark .tab-nav button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .tab-nav button.selected {
        background: #ffffff !important;
        border-bottom: 3px solid #2563eb !important;
        color: #2563eb !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.03);
    }
    .dark .tab-nav button.selected {
        background: #18181b !important;
        border-bottom: 3px solid #3b82f6 !important;
        color: #60a5fa !important;
    }

    /* Buttons Modern Look */
    button {
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        border-radius: 10px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    button.primary {
        background: linear-gradient(135deg, #18181b 0%, #27272a 100%) !important;
        color: #ffffff !important;
        border: 1px solid #3f3f46 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    button.primary:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.18) !important;
        filter: brightness(1.1);
    }
    button.secondary {
        background: #f4f4f5 !important;
        border: 1px solid #e4e4e7 !important;
        color: #18181b !important;
    }
    .dark button.secondary {
        background: #27272a !important;
        border: 1px solid #3f3f46 !important;
        color: #f4f4f5 !important;
    }
    button.secondary:hover {
        background: #e4e4e7 !important;
        transform: translateY(-1px);
    }
    button.stop {
        background: #fee2e2 !important;
        color: #dc2626 !important;
        border: 1px solid #fecaca !important;
    }
    .dark button.stop {
        background: #450a0a !important;
        color: #f87171 !important;
        border: 1px solid #7f1d1d !important;
    }
    button.stop:hover {
        background: #fecaca !important;
        transform: translateY(-1px);
    }
    
    /* Preset Chips */
    .preset-chip {
        font-size: 13px !important;
        font-weight: 500 !important;
        border-radius: 20px !important;
        padding: 5px 14px !important;
        background: #f4f4f5 !important;
        border: 1px solid #e4e4e7 !important;
        color: #27272a !important;
        cursor: pointer;
    }
    .preset-chip:hover {
        background: #18181b !important;
        color: #ffffff !important;
        border-color: #18181b !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .dark .preset-chip {
        background: #27272a !important;
        border-color: #3f3f46 !important;
        color: #e4e4e7 !important;
    }
    .dark .preset-chip:hover {
        background: #ffffff !important;
        color: #18181b !important;
        border-color: #ffffff !important;
    }

    /* UX Cards / Panels */
    .ux-card {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .dark .ux-card {
        background: #18181b;
        border: 1px solid #27272a;
    }

    /* Audio components */
    .compact-audio audio {
        height: 48px !important;
    }
    
    /* Segment Audio Card Box */
    .segment-card {
        border: 1px solid #e4e4e7;
        border-radius: 12px;
        padding: 14px 16px;
        background: #ffffff;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .segment-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .dark .segment-card {
        border: 1px solid #27272a;
        background: #18181b;
    }
    .dark .segment-card:hover {
        border-color: #3f3f46;
    }

    /* Step Badge Numbers */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        background: #18181b;
        color: #ffffff;
        border-radius: 50%;
        font-size: 13px;
        font-weight: 700;
        margin-right: 8px;
    }
    .dark .step-badge {
        background: #3b82f6;
        color: #ffffff;
    }

    /* Character Counter styling */
    .char-counter {
        font-size: 12px;
        color: #71717a;
        text-align: right;
        margin-top: 4px;
        font-weight: 500;
    }
    .dark .char-counter {
        color: #a1a1aa;
    }

    /* Toast Notification */
    #omni-toast {
        visibility: hidden;
        min-width: 250px;
        background: #18181b;
        color: #ffffff;
        text-align: center;
        border-radius: 8px;
        padding: 12px 18px;
        position: fixed;
        z-index: 9999;
        bottom: 30px;
        right: 30px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.15);
        opacity: 0;
        transition: opacity 0.3s, bottom 0.3s, visibility 0.3s;
    }
    #omni-toast.show {
        visibility: visible;
        opacity: 1;
        bottom: 40px;
    }

    /* Copy Button */
    .btn-copy-action {
        font-size: 12px !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        background: #f4f4f5 !important;
        border: 1px solid #e4e4e7 !important;
        color: #3f3f46 !important;
    }
    .dark .btn-copy-action {
        background: #27272a !important;
        border: 1px solid #3f3f46 !important;
        color: #d4d4d8 !important;
    }
    .btn-copy-action:hover {
        background: #e4e4e7 !important;
        color: #18181b !important;
    }

    /* Clean subtle scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-thumb {
        background: #d4d4d8;
        border-radius: 4px;
    }
    .dark ::-webkit-scrollbar-thumb {
        background: #3f3f46;
    }
    """
    return theme, css


def get_head_js() -> str:
    """Returns Client-side Keep-Alive JavaScript snippet for preventing Colab idle disconnections and Toast notification / Copy helpers."""
    return """
    <script>
    (function() {
        console.log("⚡ OmniVoice Colab Keep-Alive & UI Enhancements Active.");
        
        // Colab Keep-Alive Heartbeat
        setInterval(function() {
            try {
                fetch(window.location.href, { method: 'HEAD', mode: 'no-cors', cache: 'no-store' }).catch(function(){});
            } catch(e) {}
        }, 45000);

        // Toast Notification System
        window.showOmniToast = function(msg) {
            let toast = document.getElementById("omni-toast");
            if (!toast) {
                toast = document.createElement("div");
                toast.id = "omni-toast";
                document.body.appendChild(toast);
            }
            toast.innerText = msg;
            toast.className = "show";
            setTimeout(function() {
                toast.className = toast.className.replace("show", "");
            }, 3000);
        };

        // Clipboard Copy Helper
        window.copyTextToClipboard = function(text, successMsg) {
            if (!text) return;
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(text).then(function() {
                    window.showOmniToast(successMsg || "📋 Đã sao chép vào bộ nhớ tạm!");
                }).catch(function() {
                    _fallbackCopy(text, successMsg);
                });
            } else {
                _fallbackCopy(text, successMsg);
            }
        };

        function _fallbackCopy(text, successMsg) {
            var textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.left = "-999999px";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                document.execCommand('copy');
                window.showOmniToast(successMsg || "📋 Đã sao chép vào bộ nhớ tạm!");
            } catch (err) {
                console.error('Fallback copy failed', err);
            }
            document.body.removeChild(textArea);
        }
    })();
    </script>
    """
