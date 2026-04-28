import os
import json
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from google import genai
from google.genai import types

# Configure Gemini AI
GEMINI_API_KEY = "ENTER_YOUR_GEMINI_KEY"
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# Load the YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


def analyze_plant(image):
    if image is None:
        return "### No image selected.", "Please upload an image first.", "-", "-", "-"

    # Gradio image is RGB numpy array. YOLO/OpenCV pipeline uses BGR.
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model.predict(source=img_bgr, save=False, save_txt=False, conf=0.15, iou=0.45)
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        detections.append({"class_name": class_name, "confidence": round(confidence, 2)})

    gemini_disease = ""
    gemini_health_report = ""
    gemini_symptoms = ""
    gemini_treatment = ""

    if gemini_client:
        try:
            detected_names = [d["class_name"] for d in detections]
            if detected_names:
                disease_list = ", ".join(detected_names)
                prompt = (
                    f"Our YOLO model detected: {disease_list}. Verify this by examining the image. "
                    "Return a strict JSON object with four exact keys: "
                    "'detected_disease' (a concise name of the disease, or 'Healthy Plant' if none), "
                    "'health_report' (a paragraph summarizing the general health status), "
                    "'symptoms' (a bulleted list of visual symptoms in markdown), and "
                    "'treatment' (a bulleted list of what to do to cure or prevent it in markdown)."
                )
            else:
                prompt = (
                    "Our YOLO model did not detect any plant diseases. Examine the image. "
                    "Return a strict JSON object with four exact keys: "
                    "'detected_disease' (a concise name of the disease you see, or 'Healthy Plant' if none), "
                    "'health_report' (a paragraph summarizing health), "
                    "'symptoms' (bulleted list of symptoms in markdown, or 'None'), and "
                    "'treatment' (bulleted list of treatments in markdown, or 'Keep watering normally')."
                )

            model_candidates = [GEMINI_MODEL, "gemini-flash-latest"]
            model_candidates = list(dict.fromkeys(model_candidates))
            gemini_response = None
            last_error = None

            for candidate_model in model_candidates:
                try:
                    gemini_response = gemini_client.models.generate_content(
                        model=candidate_model,
                        contents=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=cv2.imencode(".jpg", img_bgr)[1].tobytes(), mime_type="image/jpeg"),
                        ],
                        config=types.GenerateContentConfig(response_mime_type="application/json"),
                    )
                    break
                except Exception as model_error:
                    last_error = model_error
                    error_text = str(model_error)
                    if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                        continue
                    raise

            if gemini_response is None and last_error is not None:
                raise last_error

            try:
                ai_data = json.loads(gemini_response.text)
                gemini_disease = ai_data.get("detected_disease", "")
                gemini_health_report = ai_data.get("health_report", "")
                gemini_symptoms = ai_data.get("symptoms", "")
                gemini_treatment = ai_data.get("treatment", "")
                if not any([gemini_disease, gemini_health_report, gemini_symptoms, gemini_treatment]):
                    gemini_health_report = "Gemini returned an empty response. Please try again with a clearer image."
            except json.JSONDecodeError:
                gemini_health_report = "Gemini returned a non-JSON response. Make sure your API key is valid and try again."

        except Exception as e:
            error_text = str(e)
            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                if detections:
                    primary_detection = detections[0]["class_name"].replace("_", " ")
                    gemini_disease = primary_detection
                    gemini_health_report = (
                        "Gemini quota limit reached right now, so this result uses YOLO-only detection. "
                        f"Detected pattern: {primary_detection}."
                    )
                    gemini_symptoms = "- Detection confidence is based on the YOLO model output."
                    gemini_treatment = "- Retry Gemini analysis after quota reset.\n- Meanwhile, isolate affected leaves and monitor spread."
                else:
                    gemini_disease = "Healthy Plant"
                    gemini_health_report = "Gemini quota limit reached right now. YOLO did not detect visible disease in this image."
                    gemini_symptoms = "- No strong disease pattern detected by YOLO."
                    gemini_treatment = "- Continue regular watering and monitoring.\n- Retry AI verification later when quota resets."
            elif "API key" in error_text or "403" in error_text or "401" in error_text:
                gemini_health_report = "Gemini authentication failed. Please verify the API key."
            else:
                gemini_health_report = "Gemini verification is temporarily unavailable. Showing YOLO result only."
    else:
        gemini_health_report = "Gemini API key is missing."

    if not gemini_disease:
        if detections:
            gemini_disease = detections[0]["class_name"].replace("_", " ")
        else:
            gemini_disease = "Healthy Plant"

    detections_text = json.dumps(detections, indent=2)
    top_det = max(detections, key=lambda d: d.get("confidence", 0)) if detections else None
    if top_det:
        top_conf = round(float(top_det.get("confidence", 0)) * 100, 1)
        conf_line = f"YOLO confidence: {top_conf}%"
    else:
        conf_line = "No YOLO detections"

    disease_md = f"### {gemini_disease}\n\n> {conf_line}"
    return (
        disease_md,
        gemini_health_report,
        gemini_symptoms or "-",
        gemini_treatment or "-",
    )


CUSTOM_CSS = """
body, .gradio-container {
    background: radial-gradient(circle at 10% 20%, #15482d 0%, #081a12 35%, #061510 100%) !important;
    color: #e8fff2 !important;
}

#app-shell {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.hero {
    text-align: center;
    padding: 14px 14px 4px 14px;
}

.logo-badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(116, 255, 177, 0.15);
    border: 1px solid rgba(116, 255, 177, 0.35);
    font-weight: 700;
    letter-spacing: .3px;
}

.hero h1 {
    margin: 14px 0 8px 0;
    font-size: 2.2rem;
    color: #ecfff3;
}

.hero p {
    margin: 0 0 8px 0;
    color: #bde6c8;
}

.panel, .result-card {
    border-radius: 18px !important;
    border: 1px solid rgba(133, 240, 178, 0.22) !important;
    background: rgba(15, 44, 31, 0.55) !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 16px 30px rgba(0, 0, 0, 0.35);
}

.scan-panel {
    position: relative;
    overflow: hidden;
}

.scan-panel::after {
    content: "";
    position: absolute;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #69ffb4, transparent);
    box-shadow: 0 0 16px #69ffb4;
    animation: scanline 2.2s linear infinite;
    pointer-events: none;
}

.scan-image {
    position: relative;
    overflow: hidden;
}

.scan-image::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 50%, rgba(105, 255, 180, 0.16) 0%, rgba(0,0,0,0) 55%);
    pointer-events: none;
    z-index: 1;
}

.scan-image::after {
    content: "";
    position: absolute;
    left: -20%;
    width: 140%;
    height: 4px;
    top: 10%;
    background: linear-gradient(90deg, transparent, rgba(105, 255, 180, 0.9), transparent);
    box-shadow: 0 0 18px rgba(105, 255, 180, 0.65);
    animation: scanline 2.2s linear infinite;
    pointer-events: none;
    z-index: 2;
}

@keyframes scanline {
    0% { top: 6%; opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { top: 94%; opacity: 0; }
}

.action-row button {
    border-radius: 12px !important;
    font-weight: 700 !important;
}

.disease-box textarea {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #e9fff0 !important;
}

.disease-banner {
    padding: 10px 14px !important;
    border-radius: 16px !important;
    border: 1px solid rgba(105, 255, 180, 0.26) !important;
    background: linear-gradient(135deg, rgba(105, 255, 180, 0.12), rgba(78, 168, 255, 0.10)) !important;
}

.result-card:hover {
    transform: translateY(-2px) !important;
    transition: transform 0.2s ease !important;
}

.result-card {
    transition: transform 0.25s ease !important;
}
"""

with gr.Blocks(title="Plant Disease AI Scanner", elem_id="app-shell") as demo:
    gr.HTML(
        """
        <div class="hero">
            <div class="logo-badge">🌿 PlantGuard AI</div>
            <h1>Plant Disease AI Scanner</h1>
            <p>Upload a leaf image and get disease name, health report, symptoms, and treatment guidance.</p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes=["panel", "scan-panel"]):
            image_input = gr.Image(type="numpy", label="Upload Leaf Image", elem_classes=["scan-image"])

    with gr.Row(elem_classes=["action-row"]):
        analyze_btn = gr.Button("Scan for Disease", variant="primary")

    with gr.Row():
        disease_output = gr.Markdown(label="Disease Name", elem_classes=["result-card", "disease-banner"])

    with gr.Row():
        with gr.Column(elem_classes=["result-card"]):
            gr.Markdown("## 🩺 Health Report")
            health_output = gr.Markdown()
        with gr.Column(elem_classes=["result-card"]):
            gr.Markdown("## 🔍 Symptoms")
            symptoms_output = gr.Markdown()
        with gr.Column(elem_classes=["result-card"]):
            gr.Markdown("## 🧪 Treatment")
            treatment_output = gr.Markdown()

    analyze_btn.click(
        fn=analyze_plant,
        inputs=[image_input],
        outputs=[disease_output, health_output, symptoms_output, treatment_output],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, css=CUSTOM_CSS)
