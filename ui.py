import gradio as gr
import os
import sys
import io
import json
import time
from logger import get_logger, setup_logging
setup_logging()
log = get_logger("ui")
from main import (run_pipeline, scan_assets, parse_campaign, load_template,
                  merge_config, render_clips)

def generate_clips_wrapper(video_input, campaign_description, template, layout="auto", progress=gr.Progress(track_tqdm=True)):
    """
    A wrapper to run the pipeline and stream progress back to the UI.
    """
    t0 = time.monotonic()
    log.info("UI request input=%.100s template=%s layout=%s campaign=%.100s", video_input, template, layout, campaign_description or "")
    if not video_input:
        log.warning("UI rejected empty input")
        yield "Please provide a video input.", "{}", [], "Pipeline not started."
        return

    # --- Preview Config ---
    try:
        available_assets = scan_assets()
        parsed_config = parse_campaign(campaign_description or "", available_assets)
        if layout and layout != "auto":
            parsed_config["layout"] = layout
            parsed_config["_layout_explicit"] = True

        final_config = parsed_config
        if template and template != "None":
            tmpl = load_template(template)
            final_config = merge_config(tmpl, parsed_config)

        config_preview = json.dumps(final_config, indent=2)
        log.info("UI config preview ok template=%s", template)
    except Exception as e:
        log.exception("UI config preview failed: %s", e)
        config_preview = f"Error generating config preview: {e}"

    yield "Starting pipeline...", config_preview, [], "Running..."

    # --- Run Pipeline & Capture Logs ---
    log_accumulator = ""
    try:
        # Use the template name 'None' as None
        effective_template = template if template != "None" else None

        # The run_pipeline function now needs to be a generator
        # For this example, we'll call it and assume it updates progress internally
        # and returns the final clips.
        log.info("UI pipeline start")
        effective_layout = None if layout in (None, "auto") else layout
        final_clips_list = run_pipeline(video_input, campaign_description, effective_template, progress,
                                        layout=effective_layout)

        output_gallery = [clip['file'] for clip in final_clips_list]
        log_accumulator = f"Pipeline completed successfully. {len(output_gallery)} clips."
        log.info("UI pipeline done clips=%d elapsed=%.1fs", len(output_gallery), time.monotonic() - t0)

        yield log_accumulator, config_preview, output_gallery, "Pipeline finished."

    except Exception as e:
        log.exception("UI pipeline failed: %s", e)
        log_accumulator = f"An unexpected error occurred: {e}"
        yield log_accumulator, config_preview, [], "Error"


with gr.Blocks() as app:
    gr.Markdown("# 🎬 Viral Clips Automator")
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Textbox(label="Video Input", placeholder="Enter a YouTube URL, Google Drive link, or local file path...")
            campaign_description = gr.Textbox(
                label="Campaign Description", 
                lines=3,
                placeholder="e.g. Add logo top right, podcast style, no music"
            )
            template_choices = ["None", "podcast_clip", "tiktok_reaction", "motivational_reel"]
            template = gr.Radio(template_choices, label="Template", value="None")
            layout = gr.Radio(["auto", "speaker_crop", "stacked_split", "center_crop", "branded_fit"],
                              label="Vertical layout (auto = face-aware crop; branded_fit only without full-screen mode)",
                              value="auto")

            generate_btn = gr.Button("🚀 Generate Clips", variant="primary")
            status_box = gr.Textbox(label="Status", value="Ready.", interactive=False)

        with gr.Column(scale=3):
            gr.Markdown("### ⚙️ Parsed Campaign Config")
            config_preview = gr.Code(label="Config Preview", language="json", value="{}")

    gr.Markdown("---")
    gr.Markdown("### 🎥 Output Clips")
    gallery = gr.Gallery(label="Generated Clips", show_label=False, elem_id="gallery", columns=4, height="auto")

    generate_btn.click(
        fn=generate_clips_wrapper,
        inputs=[video_input, campaign_description, template, layout],
        outputs=[status_box, config_preview, gallery, status_box],
        concurrency_limit=1
    )

    # ── Review & Correct tab ──────────────────────────────────────────
    with gr.Tab("Review & Correct"):
        gr.Markdown("### Ranked candidates, boundary/layout/caption fixes, selective re-render")
        with gr.Row():
            with gr.Column(scale=1):
                rev_video = gr.Textbox(label="Video ID (UUID)")
                rev_load = gr.Button("Load review data")
                rev_clips = gr.Dataframe(
                    label="Clips (clip, hook, range, refine, layout, plan v)",
                    headers=["clip", "hook", "range", "refine", "layout", "plan_v"],
                    interactive=False)
                rev_cands = gr.Dataframe(
                    label="Candidates (id, source, hook, total score, status)",
                    headers=["id", "source", "hook", "score", "status"],
                    interactive=False)
            with gr.Column(scale=1):
                rev_clip_no = gr.Number(label="Clip number", precision=0, value=1)
                rev_detail = gr.Code(label="Clip detail", language="json", value="{}")
                rev_start = gr.Textbox(label="Start (s)")
                rev_end = gr.Textbox(label="End (s)")
                rev_layout = gr.Radio(["auto", "speaker_crop", "stacked_split", "center_crop", "branded_fit"],
                                      label="Layout override", value="auto")
                gr.Markdown("Manual 9:16 crop (source pixels; authoritative over detection)")
                with gr.Row():
                    rev_roi_x = gr.Number(label="x", value=0)
                    rev_roi_y = gr.Number(label="y", value=0)
                    rev_roi_w = gr.Number(label="w", value=0)
                    rev_roi_h = gr.Number(label="h", value=0)
                with gr.Row():
                    rev_roi_t0 = gr.Textbox(label="t_start (s, optional)")
                    rev_roi_t1 = gr.Textbox(label="t_end (s, optional)")
                rev_cues = gr.Dataframe(
                    label="Caption cues (edit TEXT only, word counts fixed)",
                    headers=["start", "end", "text"],
                    interactive=True)
                with gr.Row():
                    rev_apply_bounds = gr.Button("Apply boundaries")
                    rev_apply_layout = gr.Button("Apply layout")
                with gr.Row():
                    rev_apply_caps = gr.Button("Apply captions")
                    rev_apply_crop = gr.Button("Apply manual crop")
                    rev_rerender = gr.Button("Re-render this clip", variant="primary")
                rev_status = gr.Textbox(label="Review status", interactive=False)
                rev_preview = gr.Video(label="Current final")

        def _rev_load(video_id):
            from pipeline.review import list_review_clips, list_candidates
            try:
                clips = list_review_clips((video_id or "").strip())
                rows = [[c["clip_number"], (c["hook"] or "")[:60],
                         f"{c['start_time']:.1f}-{c['end_time']:.1f}",
                         c.get("refine_status"), c.get("layout"),
                         c.get("plan_version")] for c in clips]
                cands = list_candidates((video_id or "").strip())
                crows = [[c["id"][:8], c["source"], (c.get("hook") or "")[:60],
                          round((c.get("scores") or {}).get("total", 0), 3)
                          if c.get("scores") else "",
                          c["status"]] for c in cands]
                detail = json.dumps(clips[0], indent=2, default=str) if clips else "{}"
                cues = []
                if clips:
                    from db.repositories import editplan_repo
                    plan = editplan_repo.get_latest_plan(clips[0]["id"])
                    if plan:
                        pc = plan["plan"] if isinstance(plan["plan"], dict) else {}
                        for cu in ((pc.get("captions") or {}).get("cues") or []):
                            cues.append([cu["start"], cu["end"],
                                         " ".join(w.get("display", w.get("word", ""))
                                                  for w in cu.get("words", []))])
                vid = clips[0].get("output_path") if clips else None
                return (rows, crows, detail,
                        str(clips[0]["start_time"]) if clips else "",
                        str(clips[0]["end_time"]) if clips else "",
                        cues, f"loaded {len(clips)} clips, {len(cands)} candidates",
                        vid)
            except Exception as e:
                log.exception("review load failed")
                return [], [], "{}", "", "", [], f"Error: {e}", None

        def _rev_apply_bounds(video_id, clip_no, start, end):
            from pipeline.review import list_review_clips, apply_boundary_edit
            from main import get_video_duration
            from db.repositories import video_repo
            try:
                clips = list_review_clips((video_id or "").strip())
                c = next(x for x in clips if x["clip_number"] == int(clip_no))
                v = video_repo.get_video((video_id or "").strip())
                dur = None
                if v and v.get("raw_path"):
                    dur = get_video_duration(v["raw_path"])
                rng = apply_boundary_edit(c["id"], start, end, dur)
                return f"boundaries set to {rng} (user override wins)"
            except Exception as e:
                return f"Error: {e}"

        def _rev_apply_layout(video_id, clip_no, layout):
            from pipeline.review import list_review_clips, apply_layout_override
            try:
                clips = list_review_clips((video_id or "").strip())
                c = next(x for x in clips if x["clip_number"] == int(clip_no))
                return f"layout -> {apply_layout_override(c['id'], layout)}"
            except Exception as e:
                return f"Error: {e}"

        def _rev_apply_caps(video_id, clip_no, cues):
            from pipeline.review import list_review_clips, apply_caption_text
            try:
                clips = list_review_clips((video_id or "").strip())
                c = next(x for x in clips if x["clip_number"] == int(clip_no))
                texts = [str(r[2]) for r in (cues or [])]
                n = apply_caption_text(c["id"], texts)
                return f"{n} caption cues updated (timings preserved)"
            except Exception as e:
                return f"Error: {e}"

        def _rev_apply_crop(video_id, clip_no, x, y, w, h, t0, t1):
            from pipeline.review import list_review_clips, apply_crop_override
            try:
                clips = list_review_clips((video_id or "").strip())
                c = next(x for x in clips if x["clip_number"] == int(clip_no))
                roi = apply_crop_override(
                    c["id"], float(x), float(y), float(w), float(h),
                    t_start=(t0 or None), t_end=(t1 or None))
                return (f"manual 9:16 crop set {roi} "
                        f"(detection will not override it)")
            except Exception as e:
                return f"Error: {e}"

        def _rev_rerender(video_id, clip_no):
            try:
                finals = render_clips((video_id or "").strip(),
                                      clip_numbers=[int(clip_no)])
                if finals:
                    return f"re-rendered: {finals[0]['file']}", finals[0]["file"]
                return "nothing rendered (clip rejected?)", None
            except Exception as e:
                log.exception("re-render failed")
                return f"Error: {e}", None

        rev_load.click(fn=_rev_load, inputs=[rev_video],
                       outputs=[rev_clips, rev_cands, rev_detail, rev_start,
                                rev_end, rev_cues, rev_status, rev_preview])
        rev_apply_bounds.click(fn=_rev_apply_bounds,
                               inputs=[rev_video, rev_clip_no, rev_start, rev_end],
                               outputs=[rev_status])
        rev_apply_layout.click(fn=_rev_apply_layout,
                               inputs=[rev_video, rev_clip_no, rev_layout],
                               outputs=[rev_status])
        rev_apply_caps.click(fn=_rev_apply_caps,
                             inputs=[rev_video, rev_clip_no, rev_cues],
                             outputs=[rev_status])
        rev_apply_crop.click(fn=_rev_apply_crop,
                             inputs=[rev_video, rev_clip_no, rev_roi_x,
                                     rev_roi_y, rev_roi_w, rev_roi_h,
                                     rev_roi_t0, rev_roi_t1],
                             outputs=[rev_status])
        rev_rerender.click(fn=_rev_rerender, inputs=[rev_video, rev_clip_no],
                           outputs=[rev_status, rev_preview])

if __name__ == "__main__":
    app.launch(share=False, server_port=7862, theme=gr.themes.Soft())
