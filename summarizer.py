"""Module to summarize frame-level analysis via OpenAI chat completion."""
import os
import logging
import json
from typing import List, Dict

import tiktoken
from openai import OpenAI

# Initialize OpenAI client
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_frames(results: List[Dict], config: Dict) -> Dict:
    """
    Generate a final decision based on per-frame analysis.

    :param results: List of dicts with keys 'frame', 'timestamp', 'label', 'confidence'.
    :param config: Configuration dict with 'models' and 'prompts'.
    :return: Dict with keys 'label', 'confidence', and raw 'summary' text.
    """
    model = config["models"]["summarizer"]
    prompt = config["prompts"]["summarizer"]

    # Prepare text entries for each frame
    entries = [
        f"Frame {r['frame']} at {r['timestamp']}s: {r['label']} ({r['confidence']}%)"
        for r in results
    ]

    # Setup tokenizer to estimate tokens
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    # Determine approximate max tokens for prompt content (leave buffer for response)
    max_ctx = _model_max_tokens(model)
    buffer_tokens = 500
    max_prompt_tokens = max_ctx - buffer_tokens

    # Chunk entries to fit token limit
    chunks = _chunk_entries(entries, enc, max_prompt_tokens)

    summaries = []
    for idx, chunk in enumerate(chunks):
        logging.info("Summarizing chunk %d/%d...", idx + 1, len(chunks))
        summaries.append(
            _summarize_chunk(chunk, prompt, model)
        )

    # If multiple chunk summaries, summarize those summaries
    if len(summaries) > 1:
        logging.info("Aggregating %d chunk summaries...", len(summaries))
        final_text = _summarize_chunk(summaries, prompt, model)
    else:
        final_text = summaries[0]

    label, confidence = _parse_label_confidence(final_text)
    return {"label": label, "confidence": confidence, "summary": final_text}


def _chunk_entries(entries: List[str], encoder, max_tokens: int) -> List[List[str]]:
    chunks = []
    current = []
    count = 0
    for entry in entries:
        tok = len(encoder.encode(entry))
        if count + tok > max_tokens and current:
            chunks.append(current)
            current = [entry]
            count = tok
        else:
            current.append(entry)
            count += tok
    if current:
        chunks.append(current)
    return chunks


def _summarize_chunk(items: List[str], prompt: str, model: str) -> str:
    content = "\n".join(items)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content}
    ]
    resp = _client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()


def _parse_label_confidence(text: str) -> (str, float):
    import re
    label_match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    label = label_match.group(1).capitalize() if label_match else "Unknown"
    conf_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    confidence = float(conf_match.group(1)) if conf_match else 0.0
    return label, confidence


def _model_max_tokens(model: str) -> int:
    if "gpt-4" in model:
        return 8192
    if "gpt-3.5" in model:
        return 4096
    return 2048


if __name__ == "__main__":
    import sys
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if len(sys.argv) not in (2, 3):
        print("Usage: python summarizer.py <analysis_dir> [<output_txt_path>]")
        sys.exit(1)

    analysis_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else os.path.join(analysis_dir, "final.txt")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_path = os.path.join(script_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"Config file not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r") as cf:
        cfg = yaml.safe_load(cf)

    # Load all JSON analysis files
    files = sorted(
        f for f in os.listdir(analysis_dir) if f.lower().endswith(".json")
    )
    if not files:
        print("No analysis JSON files found in", analysis_dir)
        sys.exit(1)

    results = []
    for fname in files:
        path = os.path.join(analysis_dir, fname)
        with open(path, "r") as jf:
            results.append(json.load(jf))

    final = summarize_frames(results, cfg)

    with open(output_path, "w") as outf:
        outf.write(f"Overall Decision: {final['label']}\n")
        outf.write(f"Confidence: {final['confidence']}%\n\n")
        outf.write("Full summary:\n")
        outf.write(final.get("summary", ""))

    logging.info("Summary written to %s", output_path)
    print(f"Done. Summary at {output_path}")
