#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add example voice to voices/ library
"""

import json
import os
from pathlib import Path
from datetime import datetime
import soundfile as sf

# Create voices directory
VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)

# Voice configuration
voice_name = "Example_Voice"
audio_source = r"C:\Users\PC\Downloads\audio_2025-10-10_23-22-57.ogg"
ref_text = "+Я подар+ю теб+е всел+енную - самоув+еренные слов+а. Н+о +искренние. Хот+ел показ+ать +ей вс+ё. Хот+ел подел+иться вс+ем. Хот+ел, чт+обы он+а ув+идела."

# Remove stress marks for cleaner text
ref_text_clean = ref_text.replace("+", "")

print(f"[VOICE] Adding example voice: {voice_name}")
print(f"[VOICE] Source audio: {audio_source}")

# Create voice directory
voice_dir = VOICES_DIR / voice_name
voice_dir.mkdir(parents=True, exist_ok=True)

# Load and convert audio to WAV
audio_path = voice_dir / "audio.wav"
metadata_path = voice_dir / "metadata.json"

try:
    # Convert OGG to WAV using ffmpeg
    import subprocess

    print(f"[VOICE] Converting OGG to WAV with ffmpeg...")
    result = subprocess.run([
        "ffmpeg", "-i", audio_source,
        "-ar", "24000",  # Resample to 24kHz for F5-TTS
        "-ac", "1",      # Convert to mono
        "-y",            # Overwrite output
        str(audio_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise Exception("FFmpeg conversion failed")

    print(f"[VOICE] Conversion complete")

    # Load the converted WAV to get metadata
    audio_data, sample_rate = sf.read(str(audio_path))
    duration = len(audio_data) / sample_rate

    print(f"[VOICE] Audio info: {duration:.1f}s @ {sample_rate}Hz")
    print(f"[VOICE] Saved: {audio_path}")

    # Create metadata
    metadata = {
        "ref_text": ref_text_clean,
        "ref_text_with_stress": ref_text,
        "created": datetime.now().isoformat(),
        "sample_rate": sample_rate,
        "duration": duration,
        "source": "User provided example"
    }

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[VOICE] Metadata saved: {metadata_path}")
    print(f"[VOICE] Reference text: {ref_text_clean}")
    print()
    print("SUCCESS! Example voice added to voices/Example_Voice/")
    print("You can now use it in the Gradio interface.")

except FileNotFoundError:
    print(f"ERROR: Audio file not found: {audio_source}")
    print("Please check the file path")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
