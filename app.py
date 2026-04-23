import os
import io
import warnings
import numpy as np
from flask import Flask, request, render_template, jsonify
from pydub import AudioSegment
from pydub.utils import which
from flask_cors import CORS
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers.utils import logging as hf_logging

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
hf_logging.set_verbosity_error()

# Set ffmpeg path
AudioSegment.converter = which("ffmpeg")

# Initialize Flask app
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=7)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)  # type: ignore
model.eval()

# Max audio length: 5 seconds at 16 kHz (80 000 samples).
# The processor truncates to 32 000 anyway, but capping early avoids
# wasting time decoding / resampling audio that will be thrown away.
MAX_SAMPLES = 80_000
TARGET_SR = 16_000

# Emotion labels
label_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Neutral",
    5: "Confidence",
    6: "Sadness"
}

@app.route('/')
def index():
    return render_template('index4(main).html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio_data']

    try:
        # Read the uploaded bytes directly into pydub — no temp file needed
        audio = AudioSegment.from_file(file)
        # Convert to 16 kHz mono 16-bit in memory
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1).set_sample_width(2)

        # Export to an in-memory WAV buffer and read with soundfile
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        buf.seek(0)
        waveform, _ = sf.read(buf, dtype="float32")
    except Exception as e:
        return jsonify({'error': f'Audio conversion error: {str(e)}'}), 500

    try:
        # Cap length to avoid unnecessary processing
        if len(waveform) > MAX_SAMPLES:
            waveform = waveform[:MAX_SAMPLES]

        inputs = processor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SAMPLES,
        )
        input_values = inputs['input_values'].to(device)

        with torch.no_grad():
            outputs = model(input_values)
            pred = torch.argmax(outputs.logits, dim=-1).item()
    except Exception as e:
        return jsonify({'error': f'Model inference error: {str(e)}'}), 500

    label = label_map.get(int(pred), "Unknown")
    return jsonify({'prediction': label})

# Lazy initialization for Grammar Tool to prevent slow startup
_grammar_tool = None
def get_grammar_tool():
    global _grammar_tool
    if _grammar_tool is None:
        import language_tool_python
        _grammar_tool = language_tool_python.LanguageTool('en-US')
    return _grammar_tool

@app.route('/correct_text', methods=['POST'])
def correct_text():
    try:
        data = request.json
        text = data.get("text", "")

        tool = get_grammar_tool()
        matches = tool.check(text)
        
        # language_tool_python provides a utility to correct
        import language_tool_python
        corrected = language_tool_python.utils.correct(text, matches)

        return jsonify({
            "original": text,
            "corrected": corrected
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)

