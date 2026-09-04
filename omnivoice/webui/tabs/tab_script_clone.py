import os
import time
import tempfile
import zipfile
import gradio as gr
import torch
from omnivoice.webui.config import _IS_GDRIVE, _OUTPUTS_DIR
from omnivoice.webui.profile_manager import list_voice_profiles, get_voice_profile_preview, load_voice_profile
from omnivoice.webui.components import create_lang_dropdown, create_gen_settings
from omnivoice.webui.script_parser import parse_script, analyze_script_with_gemini
from omnivoice.webui.audio_engine import _clean_gpu_memory

PAGE_SIZE = 10


def build_script_clone_tab(model, _gen):
    """Constructs the Script Clone Tab UI and internal event listeners."""
    with gr.TabItem("🎬 Sinh Giọng Kịch Bản (Script Clone)"):
        sc_page_state = gr.State(value=0)
        sc_cache_state = gr.State(value={})
        sc_temp_dir_state = gr.State(value="")

        gr.Markdown(
            """
<div style="margin-bottom: 12px;">
  <h2 style="margin: 0 0 4px 0; font-size: 20px; font-weight: 700;">🎬 Sinh Giọng Theo Kịch Bản Dài (Script Clone)</h2>
  <p style="margin: 0; color: #71717a; font-size: 14px;">Tự động phân tích ngữ cảnh cảm xúc từng câu bằng Gemini AI, sinh giọng theo đợt 10 câu, tiếp tục tiến trình không lo đứt đoạn và hỗ trợ thử lại từng câu.</p>
</div>
"""
        )

        with gr.Row():
            # Left Column: Setup & Script Input
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 🎙️ Bước 1: Chọn Giọng Mẫu Đọc Kịch Bản")
                    sc_source_type = gr.Radio(
                        choices=["🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)", "📤 Tải lên Audio mẫu mới"],
                        value="🎙️ Sử dụng Hồ sơ giọng có sẵn (.pt)" if list_voice_profiles() else "📤 Tải lên Audio mẫu mới",
                        label="Nguồn giọng đọc"
                    )

                    with gr.Group(visible=bool(list_voice_profiles())) as sc_preset_group:
                        with gr.Row():
                            sc_saved_profile = gr.Dropdown(
                                label="Chọn hồ sơ giọng",
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
                            label="Tải lên file âm thanh giọng mẫu (3–10 giây)",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        sc_ref_text = gr.Textbox(
                            label="Văn bản giọng mẫu (Tùy chọn)",
                            placeholder="Để trống nếu muốn tự động nhận diện (ASR)...",
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
                    
                    sc_lang = create_lang_dropdown("Ngôn ngữ giọng đọc (Language)")

                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### 📝 Bước 2: Nhập & Phân Tích Kịch Bản")
                    sc_script = gr.Textbox(
                        label="Kịch bản phân đoạn (Script)",
                        lines=10,
                        placeholder="[#1] THỜI GIAN: 0.0 -> 5.0\nVĂN BẢN: Nội dung câu 1...\nCẢM XÚC: Hài hước\nHƯỚNG DẪN AI: High energy intro\n------------------------------------------",
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

                    with gr.Accordion("📂 Nhập kịch bản từ file (.txt / .md)", open=False):
                        with gr.Row():
                            sc_import_raw = gr.File(label="Import Kịch bản Raw (mỗi dòng 1 câu)", file_types=[".txt", ".md"], scale=1)
                            sc_import_std = gr.File(label="Import Kịch bản Timeline Chuẩn", file_types=[".txt", ".md"], scale=1)

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

                    with gr.Accordion("🤖 Tự động phân tích cảm xúc kịch bản bằng Gemini AI", open=False):
                        with gr.Row():
                            gemini_api_key = gr.Textbox(
                                label="Gemini API Key",
                                type="password",
                                placeholder="Dán Google Gemini API Key vào đây (hoặc để trống nếu đã set ENV)...",
                                value=os.environ.get("GEMINI_API_KEY", ""),
                                scale=3
                            )
                            gemini_model = gr.Dropdown(
                                label="Model Gemini",
                                choices=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
                                value="gemini-2.5-flash",
                                scale=1
                            )
                        gemini_analyze_btn = gr.Button("✨ Phân Tích Cảm Xúc Ngữ Cảnh (Gemini Flash AI)", variant="secondary")

                    with gr.Group(visible=False) as gemini_preview_group:
                        gr.Markdown("#### 📋 Kết quả gợi ý cảm xúc từ Gemini AI:")
                        gemini_suggested_script = gr.Textbox(
                            label="Kịch bản sau khi gắn thẻ cảm xúc & chỉ dẫn AI",
                            lines=8,
                            interactive=True,
                        )
                        with gr.Row():
                            gemini_apply_btn = gr.Button("✅ Đồng ý & Áp dụng kịch bản", variant="primary", scale=2)
                            gemini_cancel_btn = gr.Button("❌ Hủy", variant="secondary", scale=1)

                    with gr.Row():
                        sc_export_btn = gr.Button("💾 Xuất File Kịch Bản Chuẩn (.txt)", size="sm", scale=1)
                        sc_export_file = gr.File(label="Tải về file", interactive=False, scale=2)

                    def _on_export_script(script_text):
                        if not script_text or not script_text.strip():
                            gr.Warning("Kịch bản hiện tại đang trống.")
                            return None
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_standard_script.txt", mode="w", encoding="utf-8")
                        tmp.write(script_text)
                        tmp.close()
                        return tmp.name

                    sc_export_btn.click(_on_export_script, inputs=[sc_script], outputs=[sc_export_file])

                    gr.Markdown("💡 **Gợi ý mẫu cảm xúc nhanh:**")
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
                            gr.Warning("Vui lòng nhập hoặc import kịch bản trước khi phân tích.")
                            return gr.update(visible=False), "", "Kịch bản trống."
                        try:
                            progress(0.2, desc="Đang gửi kịch bản đến Gemini AI...")
                            res = analyze_script_with_gemini(script_text, api_key, model_nm)
                            progress(1.0, desc="Phân tích thành công!")
                            return gr.update(visible=True), res, "✅ Gemini đã phân tích xong! Hãy kiểm tra và bấm 'Đồng ý & Áp dụng'."
                        except Exception as e:
                            gr.Warning(f"Lỗi phân tích Gemini: {e}")
                            return gr.update(visible=False), "", f"Lỗi Gemini: {e}"

                    def _on_gemini_apply(suggested_text):
                        return suggested_text, gr.update(visible=False), "✅ Đã áp dụng kịch bản có gắn thẻ cảm xúc từ Gemini!"

                    def _on_gemini_cancel():
                        return gr.update(visible=False), "Đã hủy gợi ý của Gemini."

                with gr.Group(elem_classes="ux-card"):
                    gr.Markdown("### ⚙️ Bước 3: Cài Đặt Sinh Giọng & Điều Khiển")
                    (
                        sc_ns,
                        sc_gs,
                        sc_dn,
                        sc_sp,
                        sc_du,
                        sc_pp,
                        sc_po,
                    ) = create_gen_settings()

                    sc_resume = gr.Checkbox(
                        label="🔄 Tiếp tục tiến trình (Bỏ qua câu đã tạo thành công)",
                        value=True,
                        info="Tự động giữ nguyên audio đã sinh trước đó, không tốn GPU render lại.",
                    )
                    with gr.Row():
                        sc_btn = gr.Button("▶ Sinh đợt này (10 câu)", variant="primary", scale=2)
                        sc_next_btn = gr.Button("⏭ Đợt tiếp theo", variant="secondary", scale=1)
                    sc_all_btn = gr.Button("⚡ Sinh TOÀN BỘ kịch bản", variant="primary", size="lg")

            # Right Column: Segment Audio Players & Batch ZIP Download
            with gr.Column(scale=1):
                with gr.Group(elem_classes="ux-card"):
                    with gr.Row():
                        sc_prev_view_btn = gr.Button("◀ Đợt trước", size="sm", scale=1)
                        sc_page_info = gr.Markdown("### 📑 Đang xem: Phân đoạn 1 - 10", elem_classes="text-center")
                        sc_next_view_btn = gr.Button("Đợt sau ▶", size="sm", scale=1)

                    sc_audios = []
                    sc_retries = []
                    with gr.Group():
                        for i in range(1, 11):
                            with gr.Row(elem_classes="segment-card"):
                                aud = gr.Audio(label=f"Phân đoạn #{i}", type="numpy", scale=4)
                                btn = gr.Button("🔄 Thử lại", size="sm", scale=1)
                                sc_audios.append(aud)
                                sc_retries.append(btn)
                    
                    sc_zip = gr.File(label="💾 Tải về toàn bộ file âm thanh (.ZIP)")
                    sc_parsed_markdown = gr.Markdown(label="Tóm tắt phân đoạn kịch bản")
                    sc_status = gr.Textbox(label="Trạng thái & Tiến trình trực tiếp", lines=5)

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
            start_batch_time = time.time()
            elapsed_times = []

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

                # Calculate ETA & VRAM monitor
                remaining_targets = total_targets - step_i
                if elapsed_times:
                    avg_time = sum(elapsed_times) / len(elapsed_times)
                    eta_seconds = int(avg_time * remaining_targets)
                    mins, secs = divmod(eta_seconds, 60)
                    eta_str = f" | ⏳ Còn ~{mins}m {secs}s"
                else:
                    eta_str = " | ⏳ Đang tính ETA..."

                vram_str = ""
                if torch.cuda.is_available():
                    alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    res_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                    vram_str = f" | ⚡ VRAM: {alloc_gb:.1f}GB/{res_gb:.1f}GB"

                # Show current progress and exact text being generated
                progress(
                    (step_i) / max(1, total_targets),
                    desc=f"[{step_i + 1}/{total_targets}]{eta_str}{vram_str} - #{seg_id}: {text[:28]}..."
                )

                t_seg_start = time.time()
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
                    t_seg_elapsed = time.time() - t_seg_start
                    elapsed_times.append(t_seg_elapsed)

                    all_cache[idx] = (res, stat)
                    statuses.append(f"Segment #{seg_id} [Thành công - {t_seg_elapsed:.1f}s]: {stat}")

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
            total_duration = time.time() - start_batch_time
            progress(1.0, desc=f"Hoàn tất! Tổng thời gian: {total_duration:.1f}s")
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

    return {
        "sc_source_type": sc_source_type,
        "sc_preset_group": sc_preset_group,
        "sc_custom_group": sc_custom_group,
        "sc_saved_profile": sc_saved_profile,
        "sc_preset_preview": sc_preset_preview,
    }
