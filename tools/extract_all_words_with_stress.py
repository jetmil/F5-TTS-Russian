#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Извлечение всех уникальных слов с ударениями из текста
"""

import re
from collections import defaultdict

def extract_words_with_stress(text):
    """Извлекает все слова с ударениями из текста"""

    # Удаляем markdown заголовки
    text = re.sub(r'^#.*$', '', text, flags=re.MULTILINE)

    # Находим все слова, которые содержат +
    # Слово = последовательность букв и знаков + между ними
    words = re.findall(r'[а-яёА-ЯЁa-zA-Z]+\+[а-яёА-ЯЁa-zA-Z+\-]*', text)

    return words

def analyze_stress_words(file_path):
    """Анализирует все слова с ударениями"""

    print("=" * 80)
    print("АНАЛИЗ СЛОВ С УДАРЕНИЯМИ")
    print("=" * 80)

    # Читаем файл
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Извлекаем все слова
    all_words = extract_words_with_stress(text)

    print(f"\nВсего слов с ударениями в тексте: {len(all_words)}")

    # Подсчитываем уникальные слова и их частоту
    word_frequency = defaultdict(int)
    for word in all_words:
        word_lower = word.lower()
        word_frequency[word_lower] += 1

    print(f"Уникальных слов: {len(word_frequency)}")

    # Сортируем по частоте (от самых частых к редким)
    sorted_by_freq = sorted(word_frequency.items(), key=lambda x: (-x[1], x[0]))

    # Сортируем по алфавиту
    sorted_alphabetically = sorted(word_frequency.items(), key=lambda x: x[0])

    # Сохраняем результаты

    # 1. По частоте использования
    freq_file = file_path.replace('.md', '_слова_по_частоте.txt')
    with open(freq_file, 'w', encoding='utf-8') as f:
        f.write("СЛОВА С УДАРЕНИЯМИ (по частоте использования)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Всего уникальных слов: {len(word_frequency)}\n")
        f.write(f"Всего употреблений: {len(all_words)}\n")
        f.write("=" * 80 + "\n\n")

        for word, count in sorted_by_freq:
            f.write(f"{word:<40} — {count:>4} раз(а)\n")

    print(f"\n[OK] Сохранено: {freq_file}")

    # 2. По алфавиту (удобно для проверки)
    alpha_file = file_path.replace('.md', '_слова_алфавит.txt')
    with open(alpha_file, 'w', encoding='utf-8') as f:
        f.write("СЛОВА С УДАРЕНИЯМИ (по алфавиту)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Всего уникальных слов: {len(word_frequency)}\n")
        f.write("=" * 80 + "\n\n")

        current_letter = ''
        for word, count in sorted_alphabetically:
            # Убираем знаки + для определения первой буквы
            first_letter = word.replace('+', '')[0].upper()

            if first_letter != current_letter:
                current_letter = first_letter
                f.write(f"\n--- {current_letter} ---\n\n")

            f.write(f"{word:<40} ({count} раз)\n")

    print(f"[OK] Сохранено: {alpha_file}")

    # 3. Только список слов (для быстрой проверки в словаре)
    list_file = file_path.replace('.md', '_список_слов.txt')
    with open(list_file, 'w', encoding='utf-8') as f:
        f.write("СПИСОК ВСЕХ УНИКАЛЬНЫХ СЛОВ С УДАРЕНИЯМИ\n")
        f.write("=" * 80 + "\n\n")

        for word, _ in sorted_alphabetically:
            f.write(f"{word}\n")

    print(f"[OK] Сохранено: {list_file}")

    # Статистика
    print("\n" + "=" * 80)
    print("СТАТИСТИКА")
    print("=" * 80)

    # Топ-20 самых частых слов
    print("\nТоп-20 самых частых слов:")
    print("-" * 80)
    for i, (word, count) in enumerate(sorted_by_freq[:20], 1):
        print(f"{i:2}. {word:<30} — {count:>4} раз(а)")

    # Слова с одним употреблением
    rare_words = [word for word, count in word_frequency.items() if count == 1]
    print(f"\nСлов с одним употреблением: {len(rare_words)}")

    # Средняя частота
    avg_freq = len(all_words) / len(word_frequency)
    print(f"Средняя частота слова: {avg_freq:.2f} раз(а)")

    print("\n" + "=" * 80)
    print("ГОТОВО! Проверяйте файлы:")
    print(f"  1. {freq_file}")
    print(f"  2. {alpha_file}")
    print(f"  3. {list_file}")
    print("=" * 80)

if __name__ == "__main__":
    import datetime

    # Исходный файл
    file_path = r"C:\Users\PC\карусель_FINAL.md"

    # Добавляем дату и время к выходным файлам
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Временно меняем путь для создания файлов с датой
    original_path = file_path
    file_path_with_timestamp = file_path.replace('.md', f'_{timestamp}.md')

    # Анализируем исходный файл, но сохраняем с датой
    print("=" * 80)
    print("АНАЛИЗ СЛОВ С УДАРЕНИЯМИ")
    print("=" * 80)

    # Читаем файл
    with open(original_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Извлекаем все слова
    import re
    from collections import defaultdict

    # Удаляем markdown заголовки
    text = re.sub(r'^#.*$', '', text, flags=re.MULTILINE)

    # Находим все слова с ударениями
    all_words = re.findall(r'[а-яёА-ЯЁa-zA-Z]+\+[а-яёА-ЯЁa-zA-Z+\-]*', text)

    print(f"\nВсего слов с ударениями в тексте: {len(all_words)}")

    # Подсчитываем уникальные слова
    word_frequency = defaultdict(int)
    for word in all_words:
        word_lower = word.lower()
        word_frequency[word_lower] += 1

    print(f"Уникальных слов: {len(word_frequency)}")

    # Сортируем
    sorted_by_freq = sorted(word_frequency.items(), key=lambda x: (-x[1], x[0]))
    sorted_alphabetically = sorted(word_frequency.items(), key=lambda x: x[0])

    # Создаем имена файлов с датой и временем
    base_name = original_path.replace('.md', '')
    freq_file = f"{base_name}_слова_по_частоте_{timestamp}.txt"
    alpha_file = f"{base_name}_слова_алфавит_{timestamp}.txt"
    list_file = f"{base_name}_список_слов_{timestamp}.txt"

    # 1. По частоте
    with open(freq_file, 'w', encoding='utf-8') as f:
        f.write("СЛОВА С УДАРЕНИЯМИ (по частоте использования)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Исходный файл: карусель_FINAL.md\n")
        f.write(f"Дата анализа: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Всего уникальных слов: {len(word_frequency)}\n")
        f.write(f"Всего употреблений: {len(all_words)}\n")
        f.write("=" * 80 + "\n\n")

        for word, count in sorted_by_freq:
            f.write(f"{word:<40} — {count:>4} раз(а)\n")

    print(f"\n[OK] Сохранено: {freq_file}")

    # 2. По алфавиту
    with open(alpha_file, 'w', encoding='utf-8') as f:
        f.write("СЛОВА С УДАРЕНИЯМИ (по алфавиту)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Исходный файл: карусель_FINAL.md\n")
        f.write(f"Дата анализа: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Всего уникальных слов: {len(word_frequency)}\n")
        f.write("=" * 80 + "\n\n")

        current_letter = ''
        for word, count in sorted_alphabetically:
            first_letter = word.replace('+', '')[0].upper()

            if first_letter != current_letter:
                current_letter = first_letter
                f.write(f"\n--- {current_letter} ---\n\n")

            f.write(f"{word:<40} ({count} раз)\n")

    print(f"[OK] Сохранено: {alpha_file}")

    # 3. Список слов
    with open(list_file, 'w', encoding='utf-8') as f:
        f.write("СПИСОК ВСЕХ УНИКАЛЬНЫХ СЛОВ С УДАРЕНИЯМИ\n")
        f.write("=" * 80 + "\n")
        f.write(f"Исходный файл: карусель_FINAL.md\n")
        f.write(f"Дата анализа: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for word, _ in sorted_alphabetically:
            f.write(f"{word}\n")

    print(f"[OK] Сохранено: {list_file}")

    # Статистика
    print("\n" + "=" * 80)
    print("СТАТИСТИКА")
    print("=" * 80)

    print("\nТоп-20 самых частых слов:")
    print("-" * 80)
    for i, (word, count) in enumerate(sorted_by_freq[:20], 1):
        print(f"{i:2}. {word:<30} — {count:>4} раз(а)")

    rare_words = [word for word, count in word_frequency.items() if count == 1]
    print(f"\nСлов с одним употреблением: {len(rare_words)}")

    avg_freq = len(all_words) / len(word_frequency)
    print(f"Средняя частота слова: {avg_freq:.2f} раз(а)")

    print("\n" + "=" * 80)
    print("ГОТОВО! Файлы с датой и временем:")
    print(f"  1. {freq_file}")
    print(f"  2. {alpha_file}")
    print(f"  3. {list_file}")
    print("=" * 80)
