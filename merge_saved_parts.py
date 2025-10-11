#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Быстрая склейка сохранённых аудио кусков
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
import time

def fast_merge():
    """Быстрая O(n) склейка без копирований"""

    print("=" * 60)
    print("БЫСТРАЯ СКЛЕЙКА АУДИО КУСКОВ")
    print("=" * 60)

    parts_dir = Path("outputs/audio_parts")

    if not parts_dir.exists():
        print(f"ОШИБКА: Папка {parts_dir} не найдена!")
        return

    # Находим все WAV файлы
    wav_files = sorted(parts_dir.glob("part_*.wav"))

    if not wav_files:
        print("ОШИБКА: Не найдено WAV файлов!")
        return

    print(f"\n[INFO] Найдено {len(wav_files)} кусков")
    print(f"[INFO] Подсчёт общего размера...")

    start_time = time.time()

    # Шаг 1: Подсчитываем общую длину
    total_samples = 0
    sample_rate = None

    for i, wav_file in enumerate(wav_files):
        data, sr = sf.read(wav_file)
        if sample_rate is None:
            sample_rate = sr
        total_samples += len(data)

        if (i + 1) % 500 == 0:
            print(f"   [PROGRESS] Посчитано {i + 1}/{len(wav_files)} кусков")

    print(f"[OK] Общая длина: {total_samples / sample_rate / 3600:.2f} часов")
    print(f"[INFO] Выделение памяти для {total_samples} сэмплов...")

    # Шаг 2: Создаём один большой массив (аллокация один раз!)
    final_wave = np.zeros(total_samples, dtype=np.float32)

    print(f"[OK] Память выделена: {total_samples * 4 / 1024 / 1024:.1f} MB")
    print(f"[MERGE] Копирование кусков...")

    # Шаг 3: Копируем каждый кусок один раз
    offset = 0
    for i, wav_file in enumerate(wav_files):
        data, _ = sf.read(wav_file, dtype='float32')
        chunk_len = len(data)
        final_wave[offset:offset + chunk_len] = data
        offset += chunk_len

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / len(wav_files)
            eta = elapsed / progress - elapsed
            print(f"   [PROGRESS] Склеено {i + 1}/{len(wav_files)} | "
                  f"Время: {elapsed:.1f}s | Осталось: ~{eta:.1f}s")

    print(f"[OK] Все куски склеены!")

    # Шаг 4: Сохранение
    output_path = "outputs/f5tts_full_book_merged.wav"
    print(f"[SAVE] Сохранение в {output_path}...")

    sf.write(output_path, final_wave, sample_rate)

    total_time = time.time() - start_time

    print(f"[DONE] Склейка завершена!")
    print(f"[FILE] Результат: {output_path}")
    print(f"[INFO] Длительность: {len(final_wave) / sample_rate / 3600:.2f} часов")
    print(f"[INFO] Размер: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"[TIME] Время склейки: {total_time:.1f} секунд")

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)

    return output_path

if __name__ == "__main__":
    fast_merge()
