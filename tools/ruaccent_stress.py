#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ШАГ 2: Обработка текста через RUAccent
Расставляет ударения заново с точностью 95-98%
"""

import re
import sys
from ruaccent import RUAccent

print("=" * 70)
print("ШАГ 2: ОБРАБОТКА ЧЕРЕЗ RUAccent")
print("=" * 70)

# Инициализация RUAccent
print("\nЗагружаю модель RUAccent...")
try:
    accentizer = RUAccent()
    accentizer.load(omograph_model_size='turbo', use_dictionary=True)
    print("[OK] Модель загружена успешно!")
except Exception as e:
    print(f"[ОШИБКА] Не удалось загрузить модель: {e}")
    sys.exit(1)

def process_text_with_ruaccent(text):
    """
    Обрабатывает текст через RUAccent

    Args:
        text (str): Текст БЕЗ ударений

    Returns:
        str: Текст с ударениями
    """
    lines = text.split('\n')
    result_lines = []

    total_lines = len(lines)

    for line_num, line in enumerate(lines, 1):
        # Пустые строки и заголовки пропускаем
        if not line.strip() or line.strip().startswith('#'):
            result_lines.append(line)
            continue

        # Прогресс
        if line_num % 100 == 0:
            print(f"Обработано {line_num}/{total_lines} строк ({line_num/total_lines*100:.1f}%)")

        try:
            # Обрабатываем через RUAccent
            # RUAccent возвращает текст с + перед ударной гласной
            stressed_line = accentizer.process_all(line)
            result_lines.append(stressed_line)

        except Exception as e:
            print(f"[ОШИБКА] Строка {line_num}: {e}")
            # В случае ошибки оставляем строку без изменений
            result_lines.append(line)

    return '\n'.join(result_lines)


def main():
    input_file = r"C:\Users\PC\карусель_полная_книга_ударения.md"
    output_file = r"C:\Users\PC\карусель_ruaccent.md"

    print(f"\nЧитаю: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"ОШИБКА чтения: {e}")
        sys.exit(1)

    print(f"Размер файла: {len(content)} символов")

    # Удаляем все существующие ударения
    print("\nУдаляю все существующие ударения...")
    content_no_stress = re.sub(r'\+', '', content)

    stressed_before = len(re.findall(r'\+\w+', content))
    print(f"Удалено ударений: {stressed_before}")

    # Обрабатываем через RUAccent
    print("\nОбрабатываю через RUAccent (это займёт ~10-15 минут)...")
    print("-" * 70)

    result = process_text_with_ruaccent(content_no_stress)

    # Статистика
    stressed_after = len(re.findall(r'\+\w+', result))

    print(f"\n{'=' * 70}")
    print("РЕЗУЛЬТАТ:")
    print(f"{'=' * 70}")
    print(f"Слов с ударениями: {stressed_after}")

    # Сохраняем
    print(f"\nСохраняю: {output_file}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print("[OK] Файл сохранён успешно!")
    except Exception as e:
        print(f"ОШИБКА сохранения: {e}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("ГОТОВО!")
    print(f"{'=' * 70}")
    print(f"\nОригинал (stressRNN): {input_file}")
    print(f"Очищенный: C:\\Users\\PC\\карусель_cleaned.md")
    print(f"RUAccent (новый): {output_file}")
    print("\nСледующий шаг: сравнить версии")
    print("\nФайлы для сравнения:")
    print("  1. карусель_полная_книга_ударения.md (оригинал stressRNN)")
    print("  2. карусель_cleaned.md (без ошибочных ударений)")
    print("  3. карусель_ruaccent.md (RUAccent переделал)")


if __name__ == "__main__":
    main()
