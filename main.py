import sys
import os
import json
import subprocess
import argparse
import yaml
from input.input_handler import handle_input
from campaign.campaign_parser import parse_campaign
from db.repositories import video_repo, run_repo, clip_repo
from pipeline.transcriber import transcribe_audio
from pipeline.chunker import build_chunks
from pipeline.embedder import embed_all_chunks
from pipeline.analyzer import analyze
from pipeline.similarity import find_related_segments
from pipeline.clipper import cut_clips
from pipeline.regen_srt import generate_srt_for_clip
from pipeline.fast_burn import burn_subtitles
from pipeline.improvements import add_fades, apply_logo, mix_music
from config import get_config

CONFIG = get_config()

def load_template(template_name: str) -> dict:
    path = f"campaign/templates/{template_name}.json"
    with open(path) as f:
        return json.load(f)

def merge_config(template: dict, campaign_config: dict) -> dict:
    merged = {**template, **{k: v for k, v in campaign_config.items() if v is not None}}
    return merged

def scan_assets():
    logos_path = "assets/logos"
    music_path = "assets/music"
    logos = [f for f in os.listdir(logos_path) if not f.startswith('.')] if os.path.exists(logos_path) else []
    music = [f for f in os.listdir(music_path) if not f.startswith('.')] if os.path.exists(music_path) else []
    return {"logos": logos, "music": music}

def to_sec(t):
    if isinstance(t, (int, float)): return float(t)
    if len(t.split(':')) == 2: t = '00:' + t
    p = t.split(':')
    return int(p[0])*3600 + int(p[1])*60 + float(p[2])

def should_skip_stage(video_id: str, stage: str) -> bool:
    """Return True if this stage already completed successfully for this video."""
    status = run_repo.get_stage_status(video_id, stage)
    if status == "done":
        print(f"[main] Stage '{stage}' already completed — skipping")
        return True
    return False

def run_pipeline(video_input: str, campaign_description: str = None, template: str = None, progress=None):
    # Step 1: Normalize input to local file
    if progress: progress(0.05, desc="[1/8] Normalizing input...")
    raw_video_path = handle_input(video_input)
    
    # Step 2: Insert Video into DB
    video_id = video_repo.insert_video(video_input, "Unknown Title")
    print(f"✅ Video registered in DB: {video_id}")

    # Parse campaign
    if progress: progress(0.1, desc="[2/8] Parsing campaign...")
    available_assets = scan_assets()
    campaign_config = parse_campaign(campaign_description or "", available_assets)
    if template or campaign_config.get("template"):
        tmpl_name = template or campaign_config["template"]
        if os.path.exists(f"campaign/templates/{tmpl_name}.json"):
            tmpl = load_template(tmpl_name)
            campaign_config = merge_config(tmpl, campaign_config)

    # Step 3: Transcribe
    if progress: progress(0.2, desc="[3/8] Transcribing...")
    if not should_skip_stage(video_id, "transcribe"):
        run_repo.start_stage(video_id, "transcribe")
        try:
            # Extract audio first if transcriber needs it
            audio_path = "downloads/audio.wav"
            os.makedirs("downloads", exist_ok=True)
            ffmpeg_path = os.path.expanduser("~/miniforge3/bin/ffmpeg")
            if not os.path.exists(ffmpeg_path): ffmpeg_path = "ffmpeg"
            if not os.path.exists(audio_path):
                subprocess.run([ffmpeg_path, "-i", raw_video_path, "-q:a", "0", "-map", "a", audio_path, "-y"], capture_output=True)
            
            transcript_path, transcript_result = transcribe_audio(audio_path)
            run_repo.complete_stage(video_id, "transcribe")
        except Exception as e:
            run_repo.fail_stage(video_id, "transcribe", str(e))
            raise
    else:
        transcript_path = "transcripts/transcript.json"

    # Step 4: Chunk
    if progress: progress(0.4, desc="[4/8] Chunking...")
    if not should_skip_stage(video_id, "chunk"):
        run_repo.start_stage(video_id, "chunk")
        try:
            chunks = build_chunks(transcript_path, video_id)
            run_repo.complete_stage(video_id, "chunk")
        except Exception as e:
            run_repo.fail_stage(video_id, "chunk", str(e))
            raise

    # Step 5: Embed
    if progress: progress(0.5, desc="[5/8] Embedding chunks...")
    if not should_skip_stage(video_id, "embed"):
        run_repo.start_stage(video_id, "embed")
        try:
            from db.repositories import chunk_repo
            chunks = chunk_repo.get_chunks_for_video(video_id)
            embed_all_chunks(video_id, chunks)
            run_repo.complete_stage(video_id, "embed")
        except Exception as e:
            run_repo.fail_stage(video_id, "embed", str(e))
            raise

    # Step 6: Analyze
    if progress: progress(0.6, desc="[6/8] Analyzing for clips...")
    if not should_skip_stage(video_id, "analyze"):
        run_repo.start_stage(video_id, "analyze")
        try:
            from db.repositories import chunk_repo
            chunks = chunk_repo.get_chunks_for_video(video_id)
            clips = analyze(video_id, chunks, CONFIG)
            run_repo.complete_stage(video_id, "analyze")
        except Exception as e:
            run_repo.fail_stage(video_id, "analyze", str(e))
            raise

    # Step 7: Similarity
    if progress: progress(0.7, desc="[7/8] Finding related segments...")
    if not should_skip_stage(video_id, "similarity"):
        run_repo.start_stage(video_id, "similarity")
        try:
            clips = clip_repo.get_clips_for_video(video_id)
            find_related_segments(video_id, clips, top_k=CONFIG['similarity']['top_k'], threshold=CONFIG['similarity']['threshold'])
            run_repo.complete_stage(video_id, "similarity")
        except Exception as e:
            run_repo.fail_stage(video_id, "similarity", str(e))
            raise

    # Step 8: Render
    if progress: progress(0.8, desc="[8/8] Rendering clips...")
    if not should_skip_stage(video_id, "render"):
        run_repo.start_stage(video_id, "render")
        try:
            clips = clip_repo.get_clips_for_video(video_id)
            
            # Load full transcript for SRT generation
            with open('transcripts/transcript.json', 'r') as f:
                transcript_result = json.load(f)

            os.makedirs("clips", exist_ok=True)
            os.makedirs("output", exist_ok=True)
            
            final_clips = []
            ffmpeg_path = os.path.expanduser(CONFIG.get('ffmpeg', {}).get('path', 'ffmpeg'))
            if not os.path.exists(ffmpeg_path): ffmpeg_path = "ffmpeg"

            cut_clip_results = cut_clips(raw_video_path, clips)

            for i, clip in enumerate(clips):
                n = clip['clip_number']
                base_path = f"clips/clip_{n}"
                clip_raw = f"{base_path}.mp4"
                clip_vert = f"{base_path}_vertical.mp4"
                
                # Check if clip_raw exists (cut_clips succeeded)
                if not os.path.exists(clip_raw): continue

                subprocess.run([
                    ffmpeg_path, '-y', '-i', clip_raw,
                    '-filter_complex',
                    '[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=20[bg];[0:v]scale=1080:-1[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[out]',
                    '-map', '[out]', '-map', '0:a',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac',
                    clip_vert
                ], capture_output=True)

                current_path = clip_vert
                next_path = current_path

                if campaign_config.get("subtitles", True):
                    next_path = f"{base_path}_subbed.mp4"
                    srt_path = f"clips/clip_{n}.srt"
                    srt = generate_srt_for_clip(transcript_result['segments'], to_sec(clip['start_time']), to_sec(clip['end_time']), n)
                    if srt.strip():
                        with open(srt_path, 'w') as f: f.write(srt)
                        burn_subtitles(n, clip_path=current_path, output_path=next_path)
                        current_path = next_path

                if campaign_config.get("logo"):
                    next_path = f"{base_path}_logo.mp4"
                    logo_path = campaign_config["logo"]
                    if not os.path.isabs(logo_path) and not logo_path.startswith("assets/"):
                        logo_path = f"assets/logos/{logo_path}"
                    if os.path.exists(logo_path):
                        current_path = apply_logo(current_path, next_path, logo_path, campaign_config.get("logo_position", "top-right"))
                        
                if campaign_config.get("music"):
                    next_path = f"{base_path}_music.mp4"
                    music_path = os.path.join(CONFIG.get('paths', {}).get('assets_dir', 'assets'), "music", campaign_config["music"])
                    if os.path.exists(music_path):
                        current_path = mix_music(current_path, next_path, music_path, campaign_config.get("music_volume", 0.1))

                if campaign_config.get("fade", True):
                    next_path = f"output/clip_{n}_final.mp4"
                    add_fades(current_path, next_path, fade_duration=1.0)
                else:
                    next_path = f"output/clip_{n}_final.mp4"
                    if os.path.exists(current_path):
                        os.rename(current_path, next_path)

                clip_repo.update_output_path(clip["id"], next_path)
                final_clips.append({'file': next_path})

            run_repo.complete_stage(video_id, "render")
        except Exception as e:
            run_repo.fail_stage(video_id, "render", str(e))
            raise

    # Generate final report.json from DB query
    clips = clip_repo.get_clips_for_video(video_id)
    report = []
    for c in clips:
        report.append({
            "clip_number": c["clip_number"],
            "hook": c["hook"],
            "start_time": c["start_time"],
            "end_time": c["end_time"],
            "duration": c["duration_seconds"],
            "reason": c["reason"],
            "suggested_title": c["suggested_title"],
            "output_file": c["output_path"]
        })
    with open("output/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Pipeline complete! {len(report)} clips generated.")

    return final_clips

def main():
    parser = argparse.ArgumentParser(description="Viral Clips Automator")
    parser.add_argument("video_input", nargs='?', default=None, help="URL or local path to the video")
    parser.add_argument("--campaign", help="Natural language description of the campaign", default="")
    parser.add_argument("--template", help="Name of the campaign template to use", default=None)
    
    args = parser.parse_args()
    if not args.video_input:
        print("Usage: python main.py <video_path>")
    else:
        run_pipeline(args.video_input, args.campaign, args.template)

if __name__ == "__main__":
    main()
