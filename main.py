"""Main entrypoint for CV pipeline: extract frames, analyze, summarize."""
import os
import sys
import logging
import yaml

from frame_extractor import extract_frames
from frame_analyzer import analyze_frames
from summarizer import summarize_frames


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_filepath>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"Error: file '{video_path}' not found")
        sys.exit(1)

    # Load config
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    if not os.path.isfile(config_path):
        print(f"Error: config file '{config_path}' not found")
        sys.exit(1)
    with open(config_path, "r") as cf:
        config = yaml.safe_load(cf)

    # Ensure frame_interval in config
    frame_interval = config.get("frame_interval", 10)
    config["frame_interval"] = frame_interval

    # Prepare output directories next to video
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(os.path.abspath(video_path)) or os.getcwd()
    output_dir = os.path.join(video_dir, f"{video_basename}_output")
    frames_dir = os.path.join(output_dir, "frames")
    analysis_dir = os.path.join(output_dir, "analysis")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)

    # Step 1: extract frames
    logging.info("Extracting frames every %s seconds...", frame_interval)
    frame_paths = extract_frames(video_path, frames_dir, frame_interval)
    logging.info("Extracted %d frames to '%s'", len(frame_paths), frames_dir)

    # Step 2: analyze frames
    logging.info("Analyzing frames...")
    results = analyze_frames(frame_paths, analysis_dir, config)
    logging.info("Analysis complete: results written to '%s'", analysis_dir)

    # Step 3: summarize results
    logging.info("Summarizing frame-level analyses...")
    final_result = summarize_frames(results, config)

    # Write final outcome
    final_output_path = os.path.join(output_dir, "final.txt")
    with open(final_output_path, "w") as outf:
        outf.write(f"Overall Decision: {final_result['label']}\n")
        outf.write(f"Confidence: {final_result['confidence']}%\n\n")
        outf.write("Full summary:\n")
        outf.write(final_result.get("summary", ""))

    logging.info("Pipeline complete. Final result at '%s'", final_output_path)
    print(f"Done. See final result: {final_output_path}")


if __name__ == "__main__":
    main()
