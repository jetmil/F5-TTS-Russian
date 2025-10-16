#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Комплексная проверка ударений в тексте
1. Проверка однозначных слов по словарю Зализняка
2. Выписывание омонимов с контекстом
3. Поиск слов с 2+ гласными без ударения
"""

import re
import os
from collections import defaultdict
from datetime import datetime

# Гласные буквы
VOWELS = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')

def count_vowels(word):
    """Подсчитывает количество гласных в слове (без учета +)"""
    clean_word = word.replace('+', '')
    return sum(1 for char in clean_word if char in VOWELS)

def has_stress_mark(word):
    """Проверяет, есть ли в слове знак ударения +"""
    return '+' in word

def extract_words_from_text(text):
    """Извлекает все слова из текста с их позициями"""
    # Находим все слова (с ударениями и без)
    pattern = r'[а-яёА-ЯЁ]+(?:\+[а-яёА-ЯЁ]*)*'
    words = []
    for match in re.finditer(pattern, text):
        words.append({
            'word': match.group(),
            'start': match.start(),
            'end': match.end()
        })
    return words

def get_context(text, position, context_words=7):
    """Получает контекст вокруг слова (N слов до и после)"""
    words = extract_words_from_text(text)

    # Находим индекс текущего слова
    current_idx = None
    for idx, word_info in enumerate(words):
        if word_info['start'] <= position < word_info['end']:
            current_idx = idx
            break

    if current_idx is None:
        return ""

    # Берем слова до и после
    start_idx = max(0, current_idx - context_words)
    end_idx = min(len(words), current_idx + context_words + 1)

    context_words_list = []
    for i in range(start_idx, end_idx):
        word = words[i]['word']
        if i == current_idx:
            word = f">>>{word}<<<"  # Выделяем целевое слово
        context_words_list.append(word)

    return ' '.join(context_words_list)

def load_zaliznyak_dictionary():
    """Загружает словарь Зализняка"""
    dict_path = r"C:\Users\PC\zalizniak-2010\paradigms.txt"

    if not os.path.exists(dict_path):
        print(f"[WARNING] Словарь Зализняка не найден: {dict_path}")
        return {}

    word_stresses = defaultdict(set)

    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Формат: слово или слово с ударением
            # Извлекаем слово и его ударение
            if "'" in line:  # Знак ударения в словаре Зализняка
                # Нормализуем: заменяем ' на +
                word_with_stress = line.split()[0].replace("'", "+")
                word_clean = word_with_stress.replace("+", "").lower()
                word_stresses[word_clean].add(word_with_stress.lower())

    print(f"[OK] Загружено {len(word_stresses)} слов из словаря Зализняка")
    return word_stresses

def check_stress_errors(file_path):
    """Комплексная проверка ударений"""

    print("=" * 80)
    print("КОМПЛЕКСНАЯ ПРОВЕРКА УДАРЕНИЙ")
    print("=" * 80)

    # Читаем файл
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Извлекаем все слова с позициями
    all_words = extract_words_from_text(text)

    print(f"\nВсего слов в тексте: {len(all_words)}")

    # 1. ПОИСК СЛОВ С 2+ ГЛАСНЫМИ БЕЗ УДАРЕНИЯ
    print("\n" + "=" * 80)
    print("1. СЛОВА С 2+ ГЛАСНЫМИ БЕЗ УДАРЕНИЯ")
    print("=" * 80)

    no_stress_words = []

    for word_info in all_words:
        word = word_info['word']
        vowel_count = count_vowels(word)
        has_stress = has_stress_mark(word)

        if vowel_count >= 2 and not has_stress:
            context = get_context(text, word_info['start'])
            no_stress_words.append({
                'word': word,
                'vowels': vowel_count,
                'context': context
            })

    print(f"\nНайдено слов с 2+ гласными без ударения: {len(no_stress_words)}")

    # Группируем по уникальным словам
    no_stress_unique = defaultdict(list)
    for item in no_stress_words:
        no_stress_unique[item['word'].lower()].append(item['context'])

    # Сохраняем результат
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    no_stress_file = f"C:\\Users\\PC\\карусель_СЛОВА_БЕЗ_УДАРЕНИЯ_{timestamp}.txt"

    with open(no_stress_file, 'w', encoding='utf-8') as f:
        f.write("СЛОВА С 2+ ГЛАСНЫМИ БЕЗ УДАРЕНИЯ\n")
        f.write("=" * 80 + "\n")
        f.write(f"Найдено уникальных слов: {len(no_stress_unique)}\n")
        f.write(f"Всего употреблений: {len(no_stress_words)}\n")
        f.write("=" * 80 + "\n\n")

        for word, contexts in sorted(no_stress_unique.items()):
            f.write(f"\n{word.upper()} ({len(contexts)} раз)\n")
            f.write("-" * 80 + "\n")
            for i, ctx in enumerate(contexts[:5], 1):  # Показываем первые 5 контекстов
                f.write(f"{i}. {ctx}\n")
            if len(contexts) > 5:
                f.write(f"... и ещё {len(contexts) - 5} употреблений\n")

    print(f"[OK] Сохранено: {no_stress_file}")

    # 2. ПРОВЕРКА ПО СЛОВАРЮ ЗАЛИЗНЯКА
    print("\n" + "=" * 80)
    print("2. ПРОВЕРКА ОДНОЗНАЧНЫХ СЛОВ ПО СЛОВАРЮ ЗАЛИЗНЯКА")
    print("=" * 80)

    zaliznyak_dict = load_zaliznyak_dictionary()

    stress_errors = []

    for word_info in all_words:
        word = word_info['word']

        if not has_stress_mark(word):
            continue

        word_clean = word.replace('+', '').lower()
        word_stressed = word.lower()

        if word_clean in zaliznyak_dict:
            correct_stresses = zaliznyak_dict[word_clean]

            # Если слово ОДНОЗНАЧНОЕ (только одно ударение в словаре)
            if len(correct_stresses) == 1:
                correct_stress = list(correct_stresses)[0]

                if word_stressed != correct_stress:
                    context = get_context(text, word_info['start'])
                    stress_errors.append({
                        'word': word,
                        'correct': correct_stress,
                        'context': context
                    })

    print(f"\nНайдено ошибок в однозначных словах: {len(stress_errors)}")

    errors_file = f"C:\\Users\\PC\\карусель_ОШИБКИ_УДАРЕНИЙ_{timestamp}.txt"

    with open(errors_file, 'w', encoding='utf-8') as f:
        f.write("ОШИБКИ В ОДНОЗНАЧНЫХ СЛОВАХ (проверка по словарю Зализняка)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Найдено ошибок: {len(stress_errors)}\n")
        f.write("=" * 80 + "\n\n")

        for error in stress_errors:
            f.write(f"\nОШИБКА: {error['word']} → ПРАВИЛЬНО: {error['correct']}\n")
            f.write(f"Контекст: {error['context']}\n")
            f.write("-" * 80 + "\n")

    print(f"[OK] Сохранено: {errors_file}")

    # 3. ПОИСК ОМОНИМОВ
    print("\n" + "=" * 80)
    print("3. ПОИСК ПОТЕНЦИАЛЬНЫХ ОМОНИМОВ")
    print("=" * 80)

    # Собираем все слова с разными ударениями
    word_variants = defaultdict(set)

    for word_info in all_words:
        word = word_info['word']

        if not has_stress_mark(word):
            continue

        word_clean = word.replace('+', '').lower()
        word_stressed = word.lower()

        word_variants[word_clean].add(word_stressed)

    # Находим слова с более чем одним вариантом ударения
    homonyms = {word: variants for word, variants in word_variants.items()
                if len(variants) > 1}

    print(f"\nНайдено потенциальных омонимов: {len(homonyms)}")

    # Выписываем омонимы с контекстом
    homonyms_file = f"C:\\Users\\PC\\карусель_ОМОНИМЫ_{timestamp}.txt"

    with open(homonyms_file, 'w', encoding='utf-8') as f:
        f.write("ПОТЕНЦИАЛЬНЫЕ ОМОНИМЫ (слова с разными ударениями)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Найдено слов: {len(homonyms)}\n")
        f.write("=" * 80 + "\n\n")

        for word_clean, variants in sorted(homonyms.items()):
            f.write(f"\n{word_clean.upper()} — варианты: {', '.join(sorted(variants))}\n")
            f.write("=" * 80 + "\n")

            # Для каждого варианта показываем контексты
            for variant in sorted(variants):
                f.write(f"\nВариант: {variant}\n")
                f.write("-" * 80 + "\n")

                # Находим контексты для этого варианта
                contexts = []
                for word_info in all_words:
                    if word_info['word'].lower() == variant:
                        ctx = get_context(text, word_info['start'])
                        contexts.append(ctx)

                for i, ctx in enumerate(contexts[:3], 1):  # Показываем первые 3
                    f.write(f"{i}. {ctx}\n")

                if len(contexts) > 3:
                    f.write(f"... и ещё {len(contexts) - 3} употреблений\n")

            f.write("\n")

    print(f"[OK] Сохранено: {homonyms_file}")

    # ИТОГОВАЯ СТАТИСТИКА
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    print(f"1. Слов без ударения (2+ гласных): {len(no_stress_unique)}")
    print(f"2. Ошибок в однозначных словах: {len(stress_errors)}")
    print(f"3. Потенциальных омонимов: {len(homonyms)}")
    print("\nФайлы для проверки:")
    print(f"  - {no_stress_file}")
    print(f"  - {errors_file}")
    print(f"  - {homonyms_file}")
    print("=" * 80)

if __name__ == "__main__":
    file_path = r"C:\Users\PC\карусель_FINAL.md"
    check_stress_errors(file_path)
