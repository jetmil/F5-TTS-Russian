#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест логики склейки аудио кусков
"""

import numpy as np
import soundfile as sf
import os

def test_merge_with_logging():
    """Тест склейки с логами"""

    print("=" * 60)
    print("ТЕСТ СКЛЕЙКИ АУДИО КУСКОВ")
    print("=" * 60)

    # Создаём тестовые данные
    target_sample_rate = 24000
    chunk_duration = 2  # 2 секунды на кусок
    num_chunks = 150  # Симулируем 150 кусков (чтобы было >100)

    print(f"\n[*] Генерация {num_chunks} тестовых кусков...")
    generated_waves = []

    for i in range(num_chunks):
        # Генерируем синусоиду с разной частотой для каждого куска
        t = np.linspace(0, chunk_duration, int(chunk_duration * target_sample_rate))
        frequency = 440 + i * 2  # Увеличиваем частоту с каждым куском
        wave = 0.3 * np.sin(2 * np.pi * frequency * t)
        generated_waves.append(wave)

        if (i + 1) % 50 == 0:
            print(f"   Сгенерировано {i + 1}/{num_chunks} кусков")

    print(f"[OK] Все {num_chunks} кусков сгенерированы!")

    # Тест сохранения кусков
    print(f"\n[*] Склейка {len(generated_waves)} кусков аудио...")

    if len(generated_waves) > 100:
        parts_dir = "outputs/audio_parts_test"
        os.makedirs(parts_dir, exist_ok=True)
        print(f"[SAVE] Сохранение кусков в {parts_dir}...")

        for idx, wave in enumerate(generated_waves):
            part_path = f"{parts_dir}/part_{idx:04d}.wav"
            sf.write(part_path, wave, target_sample_rate)
            if (idx + 1) % 100 == 0:
                print(f"   Сохранено {idx + 1}/{len(generated_waves)} кусков")

        print(f"[OK] Все {len(generated_waves)} кусков сохранены в {parts_dir}!")

    # Тест быстрой склейки без cross-fade
    print("\n[*] Быстрая склейка без cross-fade...")
    final_wave = np.concatenate(generated_waves)

    # Сохранение результата
    output_path = "outputs/test_merged.wav"
    os.makedirs("outputs", exist_ok=True)
    sf.write(output_path, final_wave, target_sample_rate)

    print(f"[OK] Склейка завершена!")
    print(f"[FILE] Результат сохранён: {output_path}")
    print(f"[INFO] Длительность: {len(final_wave) / target_sample_rate:.1f} секунд")
    print(f"[INFO] Размер: {len(final_wave) * 4 / 1024 / 1024:.1f} MB (float32)")

    print("\n" + "=" * 60)
    print("ТЕСТ ЗАВЕРШЁН УСПЕШНО!")
    print("=" * 60)

    return output_path

if __name__ == "__main__":
    test_merge_with_logging()
