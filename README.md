# Class Detection CV Pipeline

This Python pipeline processes a `.mov` video to determine whether a class is taking place in the footage. It:
1. Extracts frames at regular intervals using FFmpeg
2. Sends each frame (base64-encoded) to OpenAI’s `/v1/chat/completions` endpoint for per-frame analysis (Yes/No + confidence)
3. Summarizes all frame-level results into a final decision (Yes/No + confidence) using OpenAI
4. Saves intermediate JSON and final result under an output directory

## Features
- Configurable frame interval, model names, and prompts via `config.yaml`
- Standalone modules for debugging:
  - `frame_extractor.py`
  - `frame_analyzer.py`
  - `summarizer.py`
- Automatic token-based chunking to respect model context limits

## Requirements
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/) installed and on your `PATH`
- OpenAI API key in `OPENAI_API_KEY` environment variable

## Setup
```bash
# 1. Clone the repo
git clone <repo_url> && cd <repo_dir>

# 2. Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure parameters
# Edit config.yaml to set:
#   - frame_interval (seconds between frames)
#   - request_delay (seconds between OpenAI calls to throttle API usage)
#   - models.analyzer & models.summarizer (e.g. gpt-4.1-mini)
#   - prompts.analyzer & prompts.summarizer
```

## Usage

### Full pipeline (main)
```bash
python main.py path/to/video.mov
```
This creates `<video_basename>_output/` next to your video, containing:
- `frames/` (extracted images)
- `analysis/` (per-frame JSON results, with optional reflection responses)
- `final.txt` (overall decision + confidence + summary)

### Standalone modules
- **Extract frames**
  ```bash
  python frame_extractor.py input.mov output_frames_dir
  ```
- **Analyze frames**
  ```bash
  python frame_analyzer.py frames_dir analysis_dir
  ```
- **Summarize results**
  ```bash
  python summarizer.py analysis_dir [final_output.txt]
  ```

## Project Structure
```
.
├── config.yaml         # Pipeline configuration
├── frame_extractor.py  # Extracts frames via FFmpeg
├── frame_analyzer.py   # Per-frame OpenAI analysis
├── summarizer.py       # Aggregates analyses into final decision
├── main.py             # Orchestrates full pipeline
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## License
This project is provided as-is without warranty.