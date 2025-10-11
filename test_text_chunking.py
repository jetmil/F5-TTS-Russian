#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест разбивки текста - проверка что не теряются слова
"""

import re

def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def test_chunking():
    """Тест что не теряются слова при разбивке"""

    test_text = """
    Жил-был в одном городе мальчик по имени Петя. Он очень любил читать книги и мечтал стать писателем.
    Каждый день после школы Петя садился за свой письменный стол и писал рассказы. Сначала они получались короткими и простыми.
    Но со временем Петя научился создавать увлекательные истории с интересными персонажами и захватывающими приключениями.
    """.strip()

    print("=" * 60)
    print("ТЕСТ РАЗБИВКИ ТЕКСТА")
    print("=" * 60)

    print(f"\nОригинальный текст ({len(test_text)} символов):")
    print(test_text)
    print(f"\nВсего слов: {len(test_text.split())}")

    # Разбиваем на куски
    chunks = chunk_text(test_text, max_chars=80)

    print(f"\n\nРазбито на {len(chunks)} кусков:")
    print("-" * 60)

    total_words_in_chunks = 0
    merged_text = ""

    for i, chunk in enumerate(chunks, 1):
        words = len(chunk.split())
        total_words_in_chunks += words
        merged_text += " " + chunk if merged_text else chunk

        print(f"\nКусок {i} ({len(chunk)} символов, {words} слов):")
        print(f"  |{chunk}|")

    print("\n" + "=" * 60)
    print("ПРОВЕРКА")
    print("=" * 60)

    # Склеиваем куски обратно
    print(f"\nОригинальный текст: {len(test_text.split())} слов")
    print(f"После разбивки: {total_words_in_chunks} слов")

    # Убираем лишние пробелы для сравнения
    original_normalized = " ".join(test_text.split())
    merged_normalized = " ".join(merged_text.split())

    if original_normalized == merged_normalized:
        print("\nУСПЕХ: Все слова сохранены!")
    else:
        print("\nОШИБКА: Слова потеряны или изменены!")
        print(f"\nОригинал:\n{original_normalized}")
        print(f"\nПосле склейки:\n{merged_normalized}")

        # Находим разницу
        original_words = original_normalized.split()
        merged_words = merged_normalized.split()

        if len(original_words) != len(merged_words):
            print(f"\nРазница в количестве слов: {len(original_words)} vs {len(merged_words)}")

        # Ищем потерянные слова
        lost_words = []
        for i, (orig, merg) in enumerate(zip(original_words, merged_words)):
            if orig != merg:
                lost_words.append((i, orig, merg))

        if lost_words:
            print(f"\nНайдено {len(lost_words)} различий:")
            for idx, orig, merg in lost_words[:10]:  # Показываем первые 10
                print(f"  Позиция {idx}: '{orig}' -> '{merg}'")

if __name__ == "__main__":
    test_chunking()
