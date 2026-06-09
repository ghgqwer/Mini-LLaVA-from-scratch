# Mini-LLaVA с нуля (исследовательская имплементация)

Проект воспроизводит ключевые идеи LLaVA v1 (Visual Instruction Tuning) на чистом PyTorch. Фокус на мультимодальной “склейке”: датасет, маскирование лосса, MLP‑проектор, forward‑pass и полный цикл обучения. Никаких готовых LLaVA‑оберток от HF.

## Что реализовано
- Кастомный `Dataset` для COCO‑описаний с явным `<img>` токеном и корректной маской лосса (лосс только по ответу ассистента).
- Data collator: паддинг переменной длины, стак картинок и сохранение масок `labels`.
- LLaVA‑style forward: подмена `<img>` на проецированные визуальные эмбеддинги.
- Два варианта MLP‑проектора: базовый и с pixel‑unshuffle для сжатия визуальных токенов.
- Кастомный training loop: gradient accumulation, mixed precision, gradient checkpointing, сохранение чекпоинтов.
- Инференс‑пайплайн: сравнение untrained MLP vs trained MLP (с/без unshuffle) + эталонная SOTA‑модель.

## Архитектура (основная идея)

```
Image -> CLIP ViT -> image features -> MLP projector -> token-aligned embeddings
Text  -> LLM tokenizer -> token embeddings  -> replace <img> -> LLM forward
```

Замороженные модули:
- Vision encoder: CLIP ViT (openai/clip-vit-base-patch16)
- LLM: Qwen2.5-0.5B-Instruct

Обучаемый модуль:
- MLP‑проектор (vision_dim -> text_dim)

## Детали имплементации
- Шаблон промпта: `<|im_start|>user\n<img>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n`.
- Маска лосса: `labels[:len(prompt_tokens)] = -100`, чтобы исключить system/user токены.
- Вставка визуальных токенов: позиция `<img>` заменяется на визуальные эмбеддинги.
- Pixel‑unshuffle версия уменьшает число визуальных токенов до проекции.

## Конфиг обучения (в текущем ноутбуке)
- Dataset: mini COCO 2014 captions (Kaggle).
- Max text length: 128 токенов.
- Epochs: 6
- Batch size: 16
- Gradient accumulation: 2
- Optimizer: AdamW (lr=1e-3, weight_decay=0.1)
- Precision: fp16
- Чекпоинты: `checkpoints/llava_mlp_epoch_{k}.pth`
- Лог лосса: `checkpoints/loss_log.txt`

## Инференс и оценка
В inference‑ноутбуке сравниваются:
- Untrained MLP
- Trained MLP (no unshuffle)
- Trained MLP (unshuffle)
- SOTA‑baseline (Qwen2‑VL‑2B‑Instruct)

Функция `test()` прогоняет все четыре варианта и печатает описания для одного изображения. Есть runner по папке для пакетной проверки.

## Результаты (текущее состояние)
- Полный end‑to‑end пайплайн обучения и логирования; средний лосс на эпоху пишется в `checkpoints/loss_log.txt`.
- Качественная оценка: сравнение до/после обучения и против Qwen2‑VL.
- Примеры выводов формируются в ноутбуках для визуального анализа.

## Ноутбуки
- [main.ipynb](main.ipynb): датапайплайн, сборка модели, цикл обучения, чекпоинты.
- [inference.ipynb](inference.ipynb): генерация, сравнение, инференс по папке.

## Почему это интересно работодателю
- Чистая мультимодальная интеграция “с нуля” без скрытых оберток.
- Корректное маскирование лосса и выравнивание эмбеддингов.
- Практический инженеринг обучения (accumulation, mixed precision, checkpointing).
- Воспроизводимый инференс и baseline‑сравнения.

## Следующие шаги
- Добавить метрики (CIDEr/BLEU) на валидации.
- Подключить LoRA/QLoRA для более крупных LLM.
- Упаковать демо (Gradio) для живого инференса.
