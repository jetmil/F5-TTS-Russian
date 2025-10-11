#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Автоматический диктор для озвучки книг
Разбивает длинный текст на оптимальные куски для F5-TTS
"""

import os
import sys
import torch
import soundfile as sf
import numpy as np
from datetime import datetime

# Исправляем ffmpeg путь для pydub
os.environ['PATH'] = os.environ.get('PATH', '') + ';C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\bin'

import pydub
pydub.AudioSegment.converter = 'C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe'
pydub.AudioSegment.ffmpeg = 'C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe'  
pydub.AudioSegment.ffprobe = 'C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\bin\\ffprobe.exe'

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)

def split_text_for_book(text, max_chars=100):
    """
    Разбивает текст на оптимальные куски для озвучки книги
    """
    # Разбиваем по абзацам
    paragraphs = text.split('\n\n')
    chunks = []
    
    current_chunk = ""
    for paragraph in paragraphs:
        # Если абзац слишком длинный - режем по предложениям
        if len(paragraph) > max_chars:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + '. '
                else:
                    current_chunk += sentence + '. '
        else:
            # Добавляем целый абзац
            if len(current_chunk) + len(paragraph) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += '\n\n' + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_book_audio(
    book_text,
    ref_audio_path,
    ref_text,
    output_dir="book_output",
    model_path="ckpts/perfect_voice_dataset_final/model_last.pt",
    vocab_path="F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base/vocab.txt"
):
    """
    Генерирует аудио для целой книги
    """
    print("=" * 60)
    print("АВТОМАТИЧЕСКИЙ ДИКТОР ДЛЯ КНИГ")
    print("=" * 60)
    
    # Создаем папку для вывода
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем модель
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    
    vocoder = load_vocoder(device=device)
    model = load_model(
        model_cls=DiT,
        model_cfg=dict(
            dim=1024, depth=22, heads=16, ff_mult=2,
            text_dim=512, conv_layers=4,
        ),
        ckpt_path=model_path,
        vocab_file=vocab_path,
        device=device
    )
    
    # Подготавливаем референсное аудио
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
        ref_audio_path, ref_text
    )
    
    # Разбиваем текст на куски
    text_chunks = split_text_for_book(book_text, max_chars=80)
    print(f"Текст разбит на {len(text_chunks)} кусков")
    
    generated_files = []
    total_duration = 0
    
    for i, chunk in enumerate(text_chunks, 1):
        print(f"\nОбрабатывается кусок {i}/{len(text_chunks)}")
        print(f"Текст: {chunk[:50]}...")
        
        try:
            # Генерируем аудио для куска
            with torch.no_grad():
                generated_audio = infer_process(
                    ref_audio_processed,
                    ref_text_processed,
                    chunk,
                    model,
                    vocoder,
                    device=device,
                    speed=1.0
                )
            
            # Обработка результата генерации
            if isinstance(generated_audio, (list, tuple)):
                generated_audio = generated_audio[0]  # Берем основной аудио сигнал
            
            # Сохраняем кусок
            chunk_filename = f"chapter_{i:03d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            sf.write(chunk_path, generated_audio, 24000)
            generated_files.append(generated_audio)  # Сохраняем numpy array для склейки
            
            duration = len(generated_audio) / 24000
            total_duration += duration
            
            print(f"OK Сохранено: {chunk_filename} ({duration:.1f}с)")
            
        except Exception as e:
            print(f"ERROR Ошибка в куске {i}: {e}")
            continue
    
    # Склеиваем все куски в один файл
    if generated_files:
        print(f"\nСклеивание {len(generated_files)} аудио кусков...")
        
        # Склеиваем numpy arrays напрямую
        final_audio = np.concatenate(generated_files)
        
        # Сохраняем финальный файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(output_dir, f"complete_book_{timestamp}.wav")
        
        sf.write(final_path, final_audio, 24000)
        
        duration_min = len(final_audio) / 24000 / 60
        print(f"OK Книга озвучена!")
        print(f"Финальный файл: {final_path}")
        print(f"Общая длительность: {total_duration/60:.1f} минут")
        print(f"Размер файла: {duration_min:.1f} минут")
    
    return final_path if generated_files else None

if __name__ == "__main__":
    # Пример использования
    book_text = """
    Жил-был в одном городе мальчик по имени Петя. Он очень любил читать книги и мечтал стать писателем.
    
    Каждый день после школы Петя садился за свой письменный стол и писал рассказы. Сначала они получались короткими и простыми.
    
    Но со временем Петя научился создавать увлекательные истории с интересными персонажами и захватывающими приключениями.
    """
    
    # ОПТИМАЛЬНЫЕ НАСТРОЙКИ - 9 СЕКУНД АУДИО + ТОЧНЫЙ ТЕКСТ
    ref_audio = "reference_audio_9sec.wav"
    ref_text = "твои настоящие и будущие. Я могу все узнать - прошептала она? Все истории, все судьбы?"
    
    if os.path.exists(ref_audio):
        result = generate_book_audio(book_text, ref_audio, ref_text)
        if result:
            print(f"Успех! Книга сохранена: {result}")
    else:
        print(f"Референсное аудио не найдено: {ref_audio}")