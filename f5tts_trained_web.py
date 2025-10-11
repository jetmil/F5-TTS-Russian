#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
F5-TTS Trained Model Web Interface
Веб-интерфейс для обученной модели F5-TTS
"""

import os
import sys
import torch
import torchaudio
import gradio as gr
from pathlib import Path
import numpy as np
import tempfile
from datetime import datetime
import soundfile as sf
import json
import shutil

# Настройка ffmpeg пути для Gradio
os.environ['PATH'] = os.environ.get('PATH', '') + ';C:\\ffmpeg\\ffmpeg-master-latest-win64-gpl\\bin'

# Дефолтные значения
# ОПТИМАЛЬНЫЕ НАСТРОЙКИ ДЛЯ СТАБИЛЬНОЙ ГЕНЕРАЦИИ
DEFAULT_AUDIO = r"C:\Users\PC\Downloads\F5-TTS\reference_audio_9sec.wav"
DEFAULT_TEXT = "твои настоящие и будущие. Я могу все узнать - прошептала она? Все истории, все судьбы? Но не сможешь изменить ни одной - заметила Глафира. Они уже написаны. Зато не будет трагедий. Я буду знать."
DEFAULT_REF_TEXT = "твои настоящие и будущие. Я могу все узнать - прошептала она? Все истории, все судьбы?"

# Функция для безопасного чтения аудио файлов с кириллицей
def safe_audio_load(audio_path):
    """Безопасная загрузка аудио с поддержкой кириллицы"""
    try:
        # Сначала пробуем soundfile (лучше работает с путями)
        audio_data, sample_rate = sf.read(audio_path)
        audio_tensor = torch.from_numpy(audio_data).float()
        
        # Обрабатываем размерности
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_tensor.T  # soundfile возвращает (samples, channels)
            
        return audio_tensor, sample_rate
    except Exception as e:
        print(f"soundfile failed: {e}, trying torchaudio...")
        try:
            # Если soundfile не работает, пробуем torchaudio
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            return audio_tensor, sample_rate
        except Exception as e2:
            print(f"torchaudio also failed: {e2}")
            raise e2

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)

# Глобальные переменные для модели
model = None
vocoder = None
device = None
sample_rate = 24000
current_model = None

# Доступные модели
MODELS = {
    "base": {
        "name": "Базовая модель F5-TTS",
        "path": "F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base_v2/model_last.pt",
        "vocab": "F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base/vocab.txt",
        "description": "Оригинальная базовая модель F5-TTS"
    },
    "trained_200": {
        "name": "Обученная модель (200 шагов)",
        "path": "ckpts/perfect_voice_dataset_final/model_200.pt",
        "vocab": "F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base/vocab.txt",
        "description": "Ранняя стадия обучения - может быть лучше!"
    },
    "trained_400": {
        "name": "Обученная модель (400 шагов)",
        "path": "ckpts/perfect_voice_dataset_final/model_400.pt",
        "vocab": "F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base/vocab.txt",
        "description": "Средняя стадия обучения"
    },
    "trained_600": {
        "name": "Обученная модель (600 шагов)",
        "path": "ckpts/perfect_voice_dataset_final/model_600.pt",
        "vocab": "F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base/vocab.txt",
        "description": "Поздняя стадия обучения"
    },
    "trained_1000": {
        "name": "Обученная модель (1000 шагов)",
        "path": "ckpts/perfect_voice_dataset_final/model_1000.pt",
        "vocab": "F5-TTS_RUSSIAN_v2_only/F5TTS_v1_Base/vocab.txt",
        "description": "Финальная модель - может быть переобучена!"
    }
}

# Создаем папки для сохранения
os.makedirs("outputs", exist_ok=True)
os.makedirs("generated_audio", exist_ok=True)
os.makedirs("saved_configs", exist_ok=True)
os.makedirs("voices", exist_ok=True)

# Voice Management Functions
def load_saved_voices():
    """Загрузка списка сохраненных голосов из voices/"""
    voices = []
    voice_dir = Path("voices")
    if voice_dir.exists():
        for vdir in sorted(voice_dir.iterdir()):
            if vdir.is_dir():
                metadata_file = vdir / "metadata.json"
                if metadata_file.exists():
                    voices.append(vdir.name)
    return voices


def save_voice(voice_name, audio_path, ref_text, trim_start, trim_end):
    """Сохранение нового голоса с обрезкой"""
    if not voice_name or not voice_name.strip():
        return "ERROR Введите имя голоса!", gr.update()

    if not audio_path:
        return "ERROR Загрузите аудио!", gr.update()

    voice_name = voice_name.strip()
    voice_folder = Path("voices") / voice_name
    voice_folder.mkdir(parents=True, exist_ok=True)

    audio_save_path = voice_folder / "audio.wav"
    metadata_path = voice_folder / "metadata.json"

    try:
        # Загрузка аудио
        audio_data, sr = sf.read(audio_path)

        # Обрезка
        start_sample = int(trim_start * sr)
        end_sample = int(trim_end * sr)

        if end_sample > len(audio_data):
            end_sample = len(audio_data)
        if start_sample >= end_sample:
            return "ERROR Неверный диапазон обрезки!", gr.update()

        trimmed = audio_data[start_sample:end_sample]

        # Сохранение WAV
        sf.write(str(audio_save_path), trimmed, sr)

        # Метаданные
        metadata = {
            "ref_text": ref_text,
            "created": datetime.now().isoformat(),
            "sample_rate": sr,
            "duration": len(trimmed) / sr
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        message = f"OK Голос '{voice_name}' сохранен!"
        print(f"[VOICE] {message}")
        return message, gr.update(choices=load_saved_voices(), value=voice_name)

    except Exception as e:
        return f"ERROR Ошибка сохранения: {e}", gr.update()


def load_voice(voice_name):
    """Загрузка сохраненного голоса"""
    if not voice_name:
        return None, ""

    voice_folder = Path("voices") / voice_name
    audio_path = voice_folder / "audio.wav"
    metadata_path = voice_folder / "metadata.json"

    if not audio_path.exists():
        print(f"[VOICE] Аудио не найдено: {audio_path}")
        return None, ""

    ref_text = ""
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            ref_text = metadata.get("ref_text", "")

    print(f"[VOICE] Загружен голос '{voice_name}': {audio_path}")
    print(f"[VOICE] Референсный текст: {ref_text[:50]}...")

    # Возвращаем абсолютный путь
    return str(audio_path.absolute()), ref_text


def update_trim_slider_max(audio_path):
    """Обновление максимума слайдеров обрезки"""
    if not audio_path:
        return gr.update(maximum=30), gr.update(maximum=30, value=10)

    try:
        audio_data, sr = sf.read(audio_path)
        duration = len(audio_data) / sr
        return gr.update(maximum=duration, value=0), gr.update(maximum=duration, value=duration)
    except:
        return gr.update(maximum=30), gr.update(maximum=30, value=10)

def load_model_by_key(model_key="trained"):
    """Загрузка модели по ключу"""
    global model, vocoder, device, current_model

    if model_key not in MODELS:
        raise ValueError(f"Неизвестная модель: {model_key}")

    model_info = MODELS[model_key]

    print("=" * 50)
    print(f"Загрузка модели: {model_info['name']}")
    print("=" * 50)

    # Определение устройства
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Устройство: {device}")

        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
            print(f"GPU: {gpu_name} ({gpu_memory}GB)")

    # Пути к модели
    model_path = model_info["path"]
    vocab_path = model_info["vocab"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Словарь не найден: {vocab_path}")

    print(f"Загрузка модели из: {model_path}")
    print(f"Загрузка словаря из: {vocab_path}")

    # Загрузка vocoder (только если еще не загружен)
    if vocoder is None:
        print("Загрузка Vocos vocoder...")
        vocoder = load_vocoder(device=device)

    # Загрузка модели
    print("Загрузка F5-TTS модели...")
    model = load_model(
        model_cls=DiT,
        model_cfg=dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        ),
        ckpt_path=model_path,
        vocab_file=vocab_path,
        device=device
    )

    current_model = model_key
    print(f"OK Модель {model_info['name']} успешно загружена!")
    return model_info["name"]

def load_trained_model():
    """Загрузка обученной модели (обратная совместимость)"""
    return load_model_by_key("base")

def generate_speech(
    text,
    ref_audio,
    ref_text,
    speed=1.0,
    remove_silence=True,
    seed=-1
):
    """
    Генерация речи с использованием обученной модели
    """
    global model, vocoder, device
    
    if model is None:
        return None, "ERROR Модель не загружена!"
    
    try:
        # Установка seed для воспроизводимости
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Проверка входных данных
        if not text:
            return None, "ERROR Введите текст для синтеза!"
        
        if ref_audio is None:
            return None, "ERROR Загрузите референсное аудио!"
        
        # Разрешаем пустой референсный текст - пусть ASR сам определит
        if ref_text is None:
            ref_text = ""
        
        # Обработка референсного аудио
        print(f"Обработка референсного аудио: {ref_audio}")
        ref_audio_data, sample_rate_orig = safe_audio_load(ref_audio)
        
        # Конвертация в моно если стерео
        if ref_audio_data.shape[0] > 1:
            ref_audio_data = torch.mean(ref_audio_data, dim=0, keepdim=True)
        
        # Ресемплинг к 24kHz если нужно
        if sample_rate_orig != sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate_orig, sample_rate)
            ref_audio_data = resampler(ref_audio_data)
        
        # Нормализация
        ref_audio_data = ref_audio_data / torch.max(torch.abs(ref_audio_data))
        
        # Подготовка референса  
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            ref_audio,  # Передаем путь к файлу, а не numpy array
            ref_text
        )
        
        # Генерация
        print(f"Генерация речи: '{text[:50]}...'")
        
        with torch.no_grad():
            generated_audio = infer_process(
                ref_audio_processed,
                ref_text_processed,
                text,
                model,
                vocoder,
                device=device,
                speed=speed
            )
        
        # Обработка результата (может быть tuple с несколькими элементами)
        if isinstance(generated_audio, (list, tuple)):
            generated_audio = generated_audio[0]  # Берем основной аудио сигнал
        
        # Сохранение во временный файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/generated_{timestamp}.wav"
        os.makedirs("outputs", exist_ok=True)
        
        sf.write(output_path, generated_audio, sample_rate)
        
        # Удаление тишины если нужно (функция работает с файлами)
        if remove_silence:
            remove_silence_for_generated_wav(output_path)  # Обрабатывает файл напрямую
        
        model_name = MODELS.get(current_model, {}).get("name", "Неизвестная модель")

        info = f"""
OK **Успешно сгенерировано!**

**Параметры:**
- Модель: {model_name}
- Длина текста: {len(text)} символов
- Скорость: {speed}x
- Seed: {seed if seed >= 0 else 'случайный'}

**Файл сохранен:** {output_path}
"""
        
        return output_path, info
        
    except Exception as e:
        error_msg = f"ERROR **Ошибка генерации:** {str(e)}"
        print(error_msg)
        return None, error_msg

def save_config(text, ref_text, speed, remove_silence, seed, config_name):
    """Сохранение конфигурации"""
    if not config_name:
        return "Укажите название конфигурации!"
    
    config = {
        "text": text,
        "ref_text": ref_text,
        "speed": speed,
        "remove_silence": remove_silence,
        "seed": seed,
        "timestamp": datetime.now().isoformat()
    }
    
    config_path = f"saved_configs/{config_name}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return f"OK Конфигурация сохранена: {config_path}"

def load_config(config_name):
    """Загрузка конфигурации"""
    if not config_name:
        return None, None, 1.0, True, -1, "Выберите конфигурацию!"
    
    config_path = f"saved_configs/{config_name}.json"
    if not os.path.exists(config_path):
        return None, None, 1.0, True, -1, f"Конфигурация не найдена: {config_path}"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return (
            config.get("text", ""),
            config.get("ref_text", ""),
            config.get("speed", 1.0),
            config.get("remove_silence", True),
            config.get("seed", -1),
            f"OK Конфигурация загружена: {config_name}"
        )
    except Exception as e:
        return None, None, 1.0, True, -1, f"Ошибка загрузки: {e}"

def get_saved_configs():
    """Получение списка сохраненных конфигураций"""
    config_dir = Path("saved_configs")
    if not config_dir.exists():
        return []
    
    configs = []
    for config_file in config_dir.glob("*.json"):
        configs.append(config_file.stem)
    
    return sorted(configs)

def copy_to_outputs(audio_path):
    """Копирование файла в папку вывода"""
    if not audio_path:
        return "Нет аудио для копирования!"

    try:
        filename = os.path.basename(audio_path)
        dest_path = f"generated_audio/{filename}"
        shutil.copy2(audio_path, dest_path)
        return f"OK Файл скопирован: {dest_path}"
    except Exception as e:
        return f"Ошибка копирования: {e}"

def switch_model(model_key):
    """Переключение модели"""
    try:
        model_name = load_model_by_key(model_key)
        description = MODELS[model_key]["description"]
        return description, f"OK Модель переключена на: {model_name}"
    except Exception as e:
        return f"❌ Ошибка загрузки модели: {e}", f"❌ Ошибка: {e}"

def create_interface():
    """Создание Gradio интерфейса"""

    # CSS для красной кнопки генерации
    custom_css = """
    .generate-button {
        background: linear-gradient(to right, #8B0000, #DC143C) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
        box-shadow: 0 4px 6px rgba(139, 0, 0, 0.3) !important;
    }
    .generate-button:hover {
        background: linear-gradient(to right, #A52A2A, #FF4444) !important;
        box-shadow: 0 6px 8px rgba(139, 0, 0, 0.4) !important;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .generate-button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 4px rgba(139, 0, 0, 0.3) !important;
    }
    """

    with gr.Blocks(title="F5-TTS Trained Model", theme=gr.themes.Soft(), css=custom_css) as interface:
        gr.Markdown("""
        # F5-TTS Trained Model Interface
        
        **Ваша персонально обученная модель F5-TTS**
        - Обучена на 5.6 часах высококачественного аудио
        - 15 эпох тренировки на RTX 3090
        - 2519 сегментов с точной транскрипцией
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Входные параметры")

                # Библиотека голосов
                with gr.Accordion("🎤 Библиотека голосов", open=True):
                    with gr.Row():
                        voice_dropdown = gr.Dropdown(
                            label="Сохраненные голоса",
                            choices=load_saved_voices(),
                            value=None,
                            interactive=True,
                            scale=3
                        )
                        load_voice_btn = gr.Button("Загрузить", variant="secondary", scale=1)
                        refresh_voices_btn = gr.Button("Обновить", scale=1)

                    with gr.Accordion("💾 Сохранить новый голос", open=False):
                        voice_name_input = gr.Textbox(
                            label="Имя голоса",
                            placeholder="Введите имя для голоса"
                        )
                        voice_audio_input = gr.Audio(
                            label="Аудио для сохранения",
                            type="filepath"
                        )
                        with gr.Row():
                            trim_start = gr.Slider(0, 30, 0, label="Начало (сек)", step=0.1)
                            trim_end = gr.Slider(0, 30, 10, label="Конец (сек)", step=0.1)
                        voice_ref_text_input = gr.Textbox(
                            label="Текст образца",
                            lines=2,
                            placeholder="Текст, который произносится в аудио"
                        )
                        save_voice_btn = gr.Button("Сохранить голос", variant="primary")
                        save_voice_status = gr.Textbox(
                            label="Статус",
                            interactive=False,
                            show_label=False
                        )

                # Выбор модели
                model_dropdown = gr.Dropdown(
                    label="Выберите модель",
                    choices=[(info["name"], key) for key, info in MODELS.items()],
                    value="base",
                    interactive=True
                )

                # Информация о выбранной модели
                model_info = gr.Markdown(
                    value=MODELS["base"]["description"],
                    elem_id="model_info"
                )

                # Статус смены модели
                model_status = gr.Textbox(
                    label="Статус модели",
                    value="Модель готова к использованию",
                    interactive=False
                )

                # Текст для генерации
                text_input = gr.Textbox(
                    label="Текст для синтеза",
                    placeholder="Введите текст на русском языке...",
                    lines=3,
                    value=DEFAULT_TEXT
                )
                
                # Референсное аудио
                ref_audio_input = gr.Audio(
                    label="Референсное аудио (ваш голос)",
                    type="filepath",
                    elem_id="ref_audio",
                    value=DEFAULT_AUDIO if os.path.exists(DEFAULT_AUDIO) else None
                )
                
                # Текст референса
                ref_text_input = gr.Textbox(
                    label="Текст референсного аудио",
                    placeholder="Введите точный текст референсного аудио...",
                    lines=2,
                    value=DEFAULT_REF_TEXT
                )
                
                # Дополнительные параметры
                with gr.Accordion("⚙️ Дополнительные настройки", open=False):
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Скорость речи"
                    )
                    
                    remove_silence_checkbox = gr.Checkbox(
                        value=True,
                        label="Удалить тишину в начале/конце"
                    )
                    
                    seed_input = gr.Number(
                        value=-1,
                        label="Seed (для воспроизводимости, -1 = случайный)",
                        precision=0
                    )
                
                # Кнопка генерации
                generate_btn = gr.Button(
                    "Сгенерировать речь",
                    variant="primary",
                    size="lg",
                    elem_classes="generate-button"
                )
                
                # Управление конфигурациями
                with gr.Accordion("Сохранение и загрузка настроек", open=False):
                    config_name_input = gr.Textbox(
                        label="Название конфигурации",
                        placeholder="Моя_конфигурация"
                    )
                    
                    with gr.Row():
                        save_config_btn = gr.Button("Сохранить настройки", variant="secondary")
                        load_config_btn = gr.Button("Загрузить настройки", variant="secondary")
                    
                    config_dropdown = gr.Dropdown(
                        label="Выберите сохраненную конфигурацию",
                        choices=get_saved_configs(),
                        interactive=True
                    )
                    
                    config_status = gr.Textbox(
                        label="Статус",
                        interactive=False
                    )
            
            with gr.Column(scale=1):
                gr.Markdown("### Результат")
                
                # Выходное аудио
                output_audio = gr.Audio(
                    label="Сгенерированное аудио",
                    type="filepath",
                    elem_id="output_audio"
                )
                
                # Информация о генерации
                output_info = gr.Markdown(
                    value="*Здесь появится информация о генерации...*"
                )
                
                # Кнопки управления файлами
                with gr.Row():
                    copy_btn = gr.Button("Копировать в generated_audio", variant="secondary")
                    
                file_status = gr.Textbox(
                    label="Статус файла",
                    interactive=False
                )
        
        # Примеры
        gr.Markdown("### 💡 Примеры текстов")
        gr.Examples(
            examples=[
                ["Добрый день! Меня зовут Александр, и я рад приветствовать вас."],
                ["Технологии искусственного интеллекта развиваются с невероятной скоростью."],
                ["Сегодня прекрасная погода для прогулки в парке."],
                [DEFAULT_TEXT],
                ["Это демонстрация возможностей обученной модели синтеза речи."],
            ],
            inputs=text_input
        )
        
        # Обработчик генерации
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                text_input,
                ref_audio_input,
                ref_text_input,
                speed_slider,
                remove_silence_checkbox,
                seed_input
            ],
            outputs=[output_audio, output_info],
            show_progress=True
        )
        
        # Обработчики конфигураций
        save_config_btn.click(
            fn=save_config,
            inputs=[
                text_input,
                ref_text_input,
                speed_slider,
                remove_silence_checkbox,
                seed_input,
                config_name_input
            ],
            outputs=config_status
        ).then(
            fn=lambda: gr.update(choices=get_saved_configs()),
            outputs=config_dropdown
        )

        load_config_btn.click(
            fn=load_config,
            inputs=[config_dropdown],
            outputs=[
                text_input,
                ref_text_input,
                speed_slider,
                remove_silence_checkbox,
                seed_input,
                config_status
            ]
        )

        # Обработчик копирования файла
        copy_btn.click(
            fn=copy_to_outputs,
            inputs=[output_audio],
            outputs=[file_status]
        )

        # Обработчик смены модели
        model_dropdown.change(
            fn=switch_model,
            inputs=[model_dropdown],
            outputs=[model_info, model_status],
            show_progress=True
        )

        # Обработчики управления голосами
        load_voice_btn.click(
            fn=load_voice,
            inputs=[voice_dropdown],
            outputs=[ref_audio_input, ref_text_input]
        )

        refresh_voices_btn.click(
            fn=lambda: gr.update(choices=load_saved_voices()),
            outputs=[voice_dropdown]
        )

        voice_audio_input.upload(
            fn=update_trim_slider_max,
            inputs=[voice_audio_input],
            outputs=[trim_start, trim_end]
        )

        save_voice_btn.click(
            fn=save_voice,
            inputs=[voice_name_input, voice_audio_input, voice_ref_text_input, trim_start, trim_end],
            outputs=[save_voice_status, voice_dropdown]
        )

        # Информация внизу
        gr.Markdown("""
        ---
        ### Советы для лучшего качества:
        
        1. **Референсное аудио:** Используйте чистую запись вашего голоса (3-10 секунд)
        2. **Текст референса:** Укажите точный текст, который произносится в референсе
        3. **Пунктуация:** Используйте запятые и точки для естественных пауз
        4. **Скорость:** Начните с 1.0, затем подстройте под ваши предпочтения
        5. **Сохранение:** Используйте "Сохранить настройки" для запоминания конфигураций
        6. **Архив:** Копируйте результаты в generated_audio/ для постоянного хранения
        
        **Возможности интерфейса:**
        - Выбор между обученной и базовой моделью F5-TTS
        - Автозагрузка дефолтного аудио и текста
        - Сохранение и загрузка пресетов настроек
        - Копирование результатов в постоянную папку
        - Прогресс-бар генерации и примеры текстов

        **Доступные модели:**
        - **Обученная модель:** Персональная модель на вашем голосе (5.6ч, 15 эпох)
        - **Базовая модель:** Оригинальная F5-TTS модель для сравнения
        
        **GPU статус:** """ + (
            f"OK CUDA доступна ({torch.cuda.get_device_name(0)})" 
            if torch.cuda.is_available() 
            else "WARNING CPU режим (медленнее)"
        ) + f"""
        
        **Модель:** Обучена на 5.6ч аудио, 15 эпох, 2519 сегментов  
        **Папки:** outputs/ (временные), generated_audio/ (постоянные), saved_configs/ (настройки)
        """)
    
    return interface

def main():
    """Основная функция"""
    print("=" * 60)
    print("F5-TTS TRAINED MODEL WEB INTERFACE")
    print("=" * 60)
    
    # Загрузка модели
    try:
        load_trained_model()
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    # Создание и запуск интерфейса
    interface = create_interface()
    
    print("\n" + "=" * 60)
    print("Запуск веб-интерфейса...")
    print("=" * 60)
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()