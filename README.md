# Plant Disease AI Scanner

A Gradio-based web app that analyzes plant leaf images using a YOLO model and (optionally) verifies the result with Gemini to generate a health report, symptoms, and treatment guidance.

## Features

- Detects visible plant disease patterns from uploaded leaf images using YOLO.
- Returns a disease prediction and confidence context.
- Uses Gemini vision analysis for:
  - disease verification,
  - health summary,
  - symptom list,
  - treatment recommendations.
- Modern Gradio UI with custom styling.

## Project Structure

- `app.py` - Main application (model loading, inference, Gemini integration, UI).
- `requirements.txt` - Python dependencies.


## Requirements

- Python 3.10+ recommended
- A YOLO weights file named `best.pt` in the project root
- Gemini API key (optional but recommended for richer results)

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure Gemini (Recommended)

Set your Gemini model/key through environment variables before running:

```bash
set GEMINI_API_KEY=your_key_here
set GEMINI_MODEL=gemini-flash-latest
```

If Gemini is unavailable, the app still provides YOLO-only output.

## Run the App

```bash
python app.py
```

Default URL:

- `http://localhost:7860`

You can change port via:

```bash
set PORT=7861
python app.py
```

## Notes

- Ensure `best.pt` exists in the root folder, or update `MODEL_PATH` in `app.py`.
- For production, avoid hardcoding API keys in source code. Prefer environment variables.

