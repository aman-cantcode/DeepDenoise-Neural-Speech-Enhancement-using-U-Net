"""
Gradio web app — upload noisy audio, get enhanced audio back.

Run locally:
    python app.py
    → opens at http://127.0.0.1:7860

Deploy: push this repo to a HuggingFace Space (Gradio SDK).
"""

import numpy as np
import librosa
import soundfile as sf
import gradio as gr
import os
import tempfile

from enhance import load_model, enhance

WEIGHTS_PATH = "weights/unet_tf_weights.weights.h5"
SAMPLE_RATE  = 16000
MAX_SECONDS  = 120

print("Loading model...")
model = load_model(WEIGHTS_PATH)
print("Model ready.")


def process(input_audio_path):
    if input_audio_path is None:
        return None

    audio, _ = librosa.load(input_audio_path, sr=SAMPLE_RATE, mono=True)

    max_samples = MAX_SECONDS * SAMPLE_RATE
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    audio = audio.astype(np.float32)

    enhanced_audio = enhance(model, audio)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, enhanced_audio, SAMPLE_RATE)
    return temp_file.name


demo = gr.Interface(
    fn=process,
    inputs=gr.Audio(
        type="filepath",
        label="Upload noisy audio",
        sources=["upload"], 
        show_share_button=False,
    ),
    outputs=gr.Audio(
        type="filepath",
        label="Enhanced audio",
        show_share_button=False,
        show_download_button=True,
    ),
    title=" Speech Enhancement ",
    description="Upload a noisy speech recording. The U-Net model removes background noise and returns the cleaned audio.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))