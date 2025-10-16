#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест умной разбивки текста с адаптивной логикой
"""

import re

def chunk_text(text, max_chars=135):
    """
    Умная разбивка текста для озвучки с учетом интонации и времени генерации.

    СТРАТЕГИЯ:
    1. Короткие предложения (≤10 слов) → берем целиком (точки, восклицания, вопросы)
    2. Длинные предложения (>10 слов) → режем по запятым/двоеточиям/тире/точкам с запятой
    3. Ограничение: ~24 слова на чанк (16 сек * 1.5 слова/сек = безопасно для F5-TTS)

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""

    # Шаг 1: Разбиваем по концу предложения (. ! ? ;)
    sentences = re.split(r"(?<=[.!?;])\s+|(?<=[。！？；])", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Считаем количество слов в предложении
        word_count = len(sentence.split())

        # АДАПТИВНАЯ ЛОГИКА:
        # Короткие предложения (≤10 слов) → берем целиком
        if word_count <= 10:
            # Проверяем, влезает ли в текущий чанк
            if current_chunk:
                combined = current_chunk + " " + sentence
                combined_words = len(combined.split())
                combined_bytes = len(combined.encode("utf-8"))

                # Проверка: макс 24 слова И макс байт
                if combined_words <= 24 and combined_bytes <= max_chars:
                    current_chunk = combined
                else:
                    # Не влезает → сохраняем текущий чанк
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
            else:
                current_chunk = sentence

        # Длинные предложения (>10 слов) → режем по запятым/двоеточиям/тире
        else:
            # Разбиваем по запятым, двоеточиям, тире, точкам с запятой
            sub_parts = re.split(r"(?<=[,:;—–-])\s+", sentence)

            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue

                if current_chunk:
                    combined = current_chunk + " " + part
                    combined_words = len(combined.split())
                    combined_bytes = len(combined.encode("utf-8"))

                    # Проверка: макс 24 слова И макс байт
                    if combined_words <= 24 and combined_bytes <= max_chars:
                        current_chunk = combined
                    else:
                        # Не влезает → сохраняем текущий чанк
                        chunks.append(current_chunk.strip())
                        current_chunk = part
                else:
                    current_chunk = part

    # Добавляем последний чанк
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def test_smart_chunking():
    """Тест умной разбивки"""

    print("=" * 70)
    print("ТЕСТ УМНОЙ РАЗБИВКИ ТЕКСТА")
    print("=" * 70)

    # Тест 1: Короткие предложения (должны остаться целыми)
    print("\n1. КОРОТКИЕ ПРЕДЛОЖЕНИЯ (<=10 слов):")
    print("-" * 70)
    test1 = "Привет, мир! Это короткий текст. Он содержит простые фразы. Должны быть целыми."
    chunks1 = chunk_text(test1, max_chars=135)
    print(f"Текст: {test1}")
    print(f"\nКоличество чанков: {len(chunks1)}")
    for i, chunk in enumerate(chunks1, 1):
        words = len(chunk.split())
        print(f"  Чанк {i} ({words} слов): {chunk}")

    # Тест 2: Длинные предложения (должны резаться по запятым)
    print("\n2. ДЛИННЫЕ ПРЕДЛОЖЕНИЯ (>10 слов, резать по запятым):")
    print("-" * 70)
    test2 = """Однажды, когда солнце ярко светило на небе, мальчик решил отправиться в путешествие,
    чтобы найти свою мечту, преодолеть все препятствия и достичь успеха."""
    chunks2 = chunk_text(test2, max_chars=135)
    print(f"Текст: {test2.strip()}")
    print(f"\nКоличество чанков: {len(chunks2)}")
    for i, chunk in enumerate(chunks2, 1):
        words = len(chunk.split())
        print(f"  Чанк {i} ({words} слов): {chunk}")

    # Тест 3: Смешанный текст (короткие + длинные)
    print("\n3. СМЕШАННЫЙ ТЕКСТ (короткие + длинные):")
    print("-" * 70)
    test3 = """Жил-был в одном городе мальчик по имени Петя.
    Он очень любил читать книги, слушать музыку, гулять по парку и мечтать стать великим писателем.
    Каждый день после школы Петя садился за свой письменный стол и писал рассказы."""
    chunks3 = chunk_text(test3, max_chars=135)
    print(f"Текст: {test3.strip()}")
    print(f"\nКоличество чанков: {len(chunks3)}")
    for i, chunk in enumerate(chunks3, 1):
        words = len(chunk.split())
        print(f"  Чанк {i} ({words} слов): {chunk}")

    # Тест 4: Очень длинное предложение с двоеточием и тире
    print("\n4. ПРЕДЛОЖЕНИЕ С ДВОЕТОЧИЕМ И ТИРЕ:")
    print("-" * 70)
    test4 = """Вот что он узнал: книги — это окна в другие миры, знания помогают расти,
    а мечты вдохновляют на великие дела."""
    chunks4 = chunk_text(test4, max_chars=135)
    print(f"Текст: {test4.strip()}")
    print(f"\nКоличество чанков: {len(chunks4)}")
    for i, chunk in enumerate(chunks4, 1):
        words = len(chunk.split())
        print(f"  Чанк {i} ({words} слов): {chunk}")

    # Проверка лимита слов
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ЛИМИТА СЛОВ (макс 24 слова на чанк)")
    print("=" * 70)

    all_chunks = chunks1 + chunks2 + chunks3 + chunks4
    max_words = 0
    max_chunk = ""

    for chunk in all_chunks:
        words = len(chunk.split())
        if words > max_words:
            max_words = words
            max_chunk = chunk

    print(f"\nМаксимум слов в чанке: {max_words}")
    print(f"Чанк: {max_chunk[:80]}...")

    if max_words <= 24:
        print("\n[OK] УСПЕХ: Все чанки <= 24 слов (безопасно для 16 сек генерации)")
    else:
        print(f"\n[WARNING] ВНИМАНИЕ: Найден чанк с {max_words} словами (может превысить 16 сек)")

    # Статистика по всем тестам
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)
    print(f"Всего чанков: {len(all_chunks)}")
    print(f"Средняя длина: {sum(len(c.split()) for c in all_chunks) / len(all_chunks):.1f} слов")
    print(f"Минимум слов: {min(len(c.split()) for c in all_chunks)}")
    print(f"Максимум слов: {max(len(c.split()) for c in all_chunks)}")

if __name__ == "__main__":
    test_smart_chunking()
