#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Простой диктор для озвучки книг (без pydub)
"""

import os
import sys
import torch
import soundfile as sf
import numpy as np
from datetime import datetime

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
)

def split_text_smart(text, max_chars=80):
    """Умное разбиение текста для озвучки"""
    # Разбиваем по абзацам
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    
    for paragraph in paragraphs:
        # Если абзац короткий - берем целиком
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
        else:
            # Режем по предложениям
            sentences = paragraph.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) <= max_chars:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    return chunks

def generate_book_simple(
    book_text, 
    ref_audio_path="reference_audio_19sec.mp3",
    ref_text="Привет, меня зовут диктор.",
    output_dir="book_output"
):
    """Простая генерация книги"""
    
    print("=" * 50)
    print("ПРОСТОЙ ДИКТОР ДЛЯ КНИГ")
    print("=" * 50)
    
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
        ckpt_path="ckpts/perfect_voice_dataset_final/model_last.pt",
        vocab_file="F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base/vocab.txt",
        device=device
    )
    
    # Простая подготовка референса (без pydub)
    ref_audio, sr = sf.read(ref_audio_path)
    if len(ref_audio.shape) > 1:
        ref_audio = np.mean(ref_audio, axis=1)  # моно
    
    # Обрезаем до 10 секунд
    max_samples = int(10 * sr)
    if len(ref_audio) > max_samples:
        ref_audio = ref_audio[:max_samples]
    
    # Сохраняем обрезанный референс
    ref_short_path = os.path.join(output_dir, "ref_short.wav")
    sf.write(ref_short_path, ref_audio, sr)
    
    # Разбиваем текст
    text_chunks = split_text_smart(book_text, max_chars=70)
    print(f"Текст разбит на {len(text_chunks)} кусков")
    
    generated_files = []
    
    for i, chunk in enumerate(text_chunks, 1):
        print(f"\\nКусок {i}/{len(text_chunks)}: {chunk[:30]}...")
        
        try:
            # Используем F5-TTS напрямую с chunking
            with torch.no_grad():
                generated_audio = infer_process(
                    ref_short_path,  # Используем обрезанный референс
                    ref_text,
                    chunk,
                    model,
                    vocoder,
                    device=device,
                    speed=1.0
                )
            
            # Сохраняем кусок
            chunk_file = os.path.join(output_dir, f"chunk_{i:03d}.wav")
            sf.write(chunk_file, generated_audio, 24000)
            generated_files.append(generated_audio)
            
            print(f"OK Готово: {len(generated_audio)/24000:.1f}с")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Склеиваем все куски
    if generated_files:
        print(f"\\nСклеивание {len(generated_files)} кусков...")
        
        final_audio = np.concatenate(generated_files)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(output_dir, f"book_{timestamp}.wav")
        
        sf.write(final_path, final_audio, 24000)
        
        duration_min = len(final_audio) / 24000 / 60
        print(f"\\nOK Книга готова!")
        print(f"Файл: {final_path}")
        print(f"Длительность: {duration_min:.1f} минут")
        
        return final_path
    
    return None

if __name__ == "__main__":
    # Тестовый текст
    test_book = '''
Глава 1. Начало приключений

Жил-был в одном городе мальчик по имени Петя. Он очень любил читать книги и мечтал стать писателем.

Каждый день после школы Петя садился за свой письменный стол и писал рассказы. Сначала они получались короткими и простыми.

Но со временем Петя научился создавать увлекательные истории с интересными персонажами и захватывающими приключениями.

Глава 2. Волшебная находка

Однажды, гуляя по старому парку, Петя нашел необычную книгу. Обложка была покрыта странными символами.

Когда мальчик открыл книгу, из неё вылетели светящиеся буквы и закружились в воздухе.

"Добро пожаловать в мир магии слов!" - прозвучал голос ниоткуда.
'''
    
    result = generate_book_simple(test_book)
    if result:
        print(f"\\nУСПЕХ! Книга озвучена: {result}")
    else:
        print("\\nОШИБКА при озвучке")