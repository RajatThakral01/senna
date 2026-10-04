import sys
from main import run_pipeline, should_skip_stage, CONFIG
from db.repositories import run_repo, chunk_repo, clip_repo
from pipeline.analyzer import analyze
from pipeline.similarity import find_related_segments
from pipeline.clipper import cut_clips
from pipeline.regen_srt import generate_srt_for_clip
from pipeline.fast_burn import burn_subtitles
from pipeline.improvements import add_fades, apply_logo, mix_music
from campaign.campaign_parser import parse_campaign
import os, subprocess, json

video_id = "fdffb6a3-f5e1-47d1-b646-6c12876ebe52"
video_input = "https://youtu.be/HAnw168huqA"
raw_video_path = "input/raw_video.mp4"

print(f"Resuming pipeline for video_id {video_id}")

def to_sec(t):
    if isinstance(t, (int, float)): return float(t)
    if len(t.split(':')) == 2: t = '00:' + t
    p = t.split(':')
    return int(p[0])*3600 + int(p[1])*60 + float(p[2])

# Step 6: Analyze
if not should_skip_stage(video_id, "analyze"):
    run_repo.start_stage(video_id, "analyze")
    chunks = chunk_repo.get_chunks_for_video(video_id)
    # We will resume analyzer
    clips = analyze(video_id, chunks, CONFIG)
    run_repo.complete_stage(video_id, "analyze")

# Step 7: Similarity
if not should_skip_stage(video_id, "similarity"):
    run_repo.start_stage(video_id, "similarity")
    clips = clip_repo.get_clips_for_video(video_id)
    find_related_segments(video_id, clips, top_k=CONFIG['similarity']['top_k'], threshold=CONFIG['similarity']['threshold'])
    run_repo.complete_stage(video_id, "similarity")

# Step 8: Render
if not should_skip_stage(video_id, "render"):
    run_repo.start_stage(video_id, "render")
    clips = clip_repo.get_clips_for_video(video_id)
    with open('transcripts/transcript.json', 'r') as f:
        transcript_result = json.load(f)
    os.makedirs("clips", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    ffmpeg_path = os.path.expanduser(CONFIG.get('ffmpeg', {}).get('path', 'ffmpeg'))
    if not os.path.exists(ffmpeg_path): ffmpeg_path = "ffmpeg"
    cut_clip_results = cut_clips(raw_video_path, clips)

    for i, clip in enumerate(clips):
        n = clip['clip_number']
        base_path = f"clips/clip_{n}"
        clip_raw = f"{base_path}.mp4"
        clip_vert = f"{base_path}_vertical.mp4"
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
        next_path = f"{base_path}_subbed.mp4"
        srt_path = f"clips/clip_{n}.srt"
        srt = generate_srt_for_clip(transcript_result['segments'], to_sec(clip['start_time']), to_sec(clip['end_time']), n)
        if srt.strip():
            with open(srt_path, 'w') as f: f.write(srt)
            burn_subtitles(n, clip_path=current_path, output_path=next_path)
            current_path = next_path
        
        final_path = f"output/clip_{n}_final.mp4"
        add_fades(current_path, final_path, fade_duration=1.0)
        clip_repo.update_output_path(clip["id"], final_path)

    run_repo.complete_stage(video_id, "render")

print("Pipeline resumed and completed successfully!")
