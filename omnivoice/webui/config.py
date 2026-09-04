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
    """
    gdrive_base = "/content/drive/MyDrive/OmniVoice_Studio"
    if os.path.exists("/content/drive/MyDrive"):
        voices_dir = os.path.join(gdrive_base, "saved_voices")
        out_dir = os.path.join(gdrive_base, "outputs")
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
# Voice Design instruction templates
# ---------------------------------------------------------------------------
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
    return theme, css
