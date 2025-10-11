#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест целостности генерации - проверка что не теряются слова
"""

import os
import sys
import torch
import soundfile as sf
import numpy as np

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
)

def test_generation():
    """Тест что все слова озвучиваются"""

    print("=" * 60)
    print("ТЕСТ ЦЕЛОСТНОСТИ ГЕНЕРАЦИИ")
    print("=" * 60)

    # Загружаем модель
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nУстройство: {device}")

    vocoder = load_vocoder(device=device)
    model = load_model(
        model_cls=DiT,
        model_cfg=dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, conv_layers=4,
        ),
        ckpt_path="ckpts/perfect_voice_dataset_final/model_last.pt",
        vocab_file="F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base/vocab.txt",
        device=device
    )

    # Референсное аудио
    ref_audio_path = "reference_audio_19sec.mp3"
    ref_text = "Привет, меня зовут диктор."

    # Тестовый текст с ЯВНО РАЗЛИЧИМЫМИ словами
    test_text = "Один. Два. Три. Четыре. Пять. Шесть. Семь. Восемь. Девять. Десять."

    print(f"\nТестовый текст: {test_text}")
    print(f"Всего слов: {len(test_text.split())}")

    print("\nГенерация аудио...")

    try:
        with torch.no_grad():
            generated_audio = infer_process(
                ref_audio_path,
                ref_text,
                test_text,
                model,
                vocoder,
                device=device,
                speed=1.0
            )

        # Сохраняем
        output_path = "outputs/test_integrity.wav"
        os.makedirs("outputs", exist_ok=True)
        sf.write(output_path, generated_audio, 24000)

        duration = len(generated_audio) / 24000
        print(f"\nГотово!")
        print(f"Файл: {output_path}")
        print(f"Длительность: {duration:.1f}с")

        print("\nПРОВЕРКА:")
        print("1. Прослушай файл outputs/test_integrity.wav")
        print("2. Убедись что все 10 слов озвучены")
        print("3. Если какое-то слово пропущено - проблема в модели/vocoder")

    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation()
