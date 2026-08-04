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

from model.unet       import build_unet
from audio.stft_utils import wav_to_mag_phase, mag_phase_to_wav


def load_model(weights_path):
    model = build_unet()
    dummy = tf.zeros([1, 257, 501, 1], dtype=tf.float32)
    model(dummy, training=False)
    model.load_weights(weights_path)
    print(f"  Model loaded from: {weights_path}")
    return model


def enhance(model, noisy_wav):
    original_len = len(noisy_wav)
    noisy_tensor = tf.constant(noisy_wav[np.newaxis, :], dtype=tf.float32)
    mag, phase = wav_to_mag_phase(noisy_tensor)
    mag_input = tf.expand_dims(mag, axis=-1)
    enhanced_mag = model(mag_input, training=False)
    enhanced_mag = tf.squeeze(enhanced_mag, axis=-1)
    enhanced_wav = mag_phase_to_wav(enhanced_mag, phase, target_len=original_len)
    return enhanced_wav[0].numpy()

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