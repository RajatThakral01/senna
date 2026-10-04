import gradio as gr
import os
import sys
import io
import json
import time
from logger import get_logger, setup_logging
setup_logging()
log = get_logger("ui")
from main import run_pipeline, scan_assets, parse_campaign, load_template, merge_config

def generate_clips_wrapper(video_input, campaign_description, template, progress=gr.Progress(track_tqdm=True)):
    """
    A wrapper to run the pipeline and stream progress back to the UI.
    """
    t0 = time.monotonic()
    log.info("UI request input=%.100s template=%s campaign=%.100s", video_input, template, campaign_description or "")
    if not video_input:
        log.warning("UI rejected empty input")
        yield "Please provide a video input.", "{}", [], "Pipeline not started."
        return

    # --- Preview Config ---
    try:
        available_assets = scan_assets()
        parsed_config = parse_campaign(campaign_description or "", available_assets)

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
        final_clips_list = run_pipeline(video_input, campaign_description, effective_template, progress)

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
        inputs=[video_input, campaign_description, template],
        outputs=[status_box, config_preview, gallery, status_box],
        concurrency_limit=1
    )

if __name__ == "__main__":
    app.launch(share=False, server_port=7862, theme=gr.themes.Soft())
