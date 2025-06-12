"""Module to analyze frames via OpenAI chat completion."""
import os
import re
import json
import base64
import logging
from typing import List, Dict

import openai


def analyze_frames(frame_paths: List[str], analysis_dir: str, config: Dict) -> List[Dict]:
    """
    Analyze each frame image and save per-frame JSON results.

    :param frame_paths: List of image file paths.
    :param analysis_dir: Directory where JSON analysis files will be saved.
    :param config: Configuration dict with 'models', 'prompts', and 'frame_interval'.
    :return: List of analysis results dicts.
    """
    os.makedirs(analysis_dir, exist_ok=True)
    model = config["models"]["analyzer"]
    prompt = config["prompts"]["analyzer"]
    interval = config["frame_interval"]

    results = []
    for idx, frame_path in enumerate(frame_paths):
        with open(frame_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is a base64-encoded image:\n{b64}"}
        ]
        logging.info("Analyzing frame %d/%d: %s", idx + 1, len(frame_paths), frame_path)
        resp = openai.ChatCompletion.create(model=model, messages=messages)
        content = resp.choices[0].message.content.strip()
        label, confidence = _parse_label_confidence(content)
        timestamp = idx * interval
        result = {
            "frame": os.path.basename(frame_path),
            "timestamp": timestamp,
            "label": label,
            "confidence": confidence
        }
        results.append(result)
        out_path = os.path.join(analysis_dir, f"{result['frame']}.json")
        with open(out_path, "w") as jf:
            json.dump(result, jf)

    return results


def _parse_label_confidence(text: str) -> (str, float):
    """
    Parse model response to extract a yes/no label and confidence percentage.

    :param text: The model output string.
    :return: Tuple of (label, confidence).
    """
    label_match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    label = label_match.group(1).capitalize() if label_match else "Unknown"
    conf_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    confidence = float(conf_match.group(1)) if conf_match else 0.0
    return label, confidence


if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if len(sys.argv) != 3:
        print("Usage: python frame_analyzer.py <frames_dir> <analysis_dir>")
        sys.exit(1)

    frames_dir = sys.argv[1]
    analysis_dir = sys.argv[2]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_path = os.path.join(script_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r") as cf:
        cfg = yaml.safe_load(cf)
    # build list of image files
    frame_files = sorted(
        [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    if not frame_files:
        print("No frames found in", frames_dir)
        sys.exit(1)

    analyze_frames(frame_files, analysis_dir, cfg)
    print(f"Analysis complete. JSON results in {analysis_dir}")
