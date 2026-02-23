# Установка и использование uv

## Установка uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# или через brew
brew install uv
```

После установки перезапусти терминал или:

```bash
source ~/.bashrc  # или ~/.zshrc
```

## Настройка проекта

```bash
git clone <repo-url> vlm-safety
cd vlm-safety
```

### Установка зависимостей

```bash
uv sync
```

Это создаст `.venv/` в текущей директории и установит все пакеты туда. Ничего глобально не ставится.

Если есть проблемы с SSL (корпоративный прокси):

```bash
uv sync --native-tls
```

### Если нужен конкретный Python

uv сам скачает нужную версию Python:

```bash
uv python install 3.11
uv venv --python 3.11
uv sync
```

## Запуск скриптов

Всегда через `uv run` — он автоматически использует `.venv/` из текущей директории:

```bash
# Запуск скрипта
uv run python scripts/run_batch.py --help

# Запуск тестов
uv run pytest

# Любая команда в контексте venv
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Добавление пакетов

```bash
# Обычная зависимость
uv add requests

# Dev-зависимость
uv add --dev ipython

# С native TLS (если проблемы с SSL)
uv add --native-tls some-package
```

## Структура

После `uv sync` в проекте появится:

```
vlm-safety/
├── .venv/            # виртуальное окружение (в .gitignore)
├── pyproject.toml    # зависимости проекта
└── uv.lock           # lock-файл (коммитится в git)
```

Всё изолировано в `.venv/` внутри проекта. Никаких глобальных пакетов.

## GPU-сервер: быстрый старт

```bash
# 1. Установить uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Склонировать и поставить зависимости
git clone <repo-url> vlm-safety
cd vlm-safety
uv sync

# 3. Скачать CLIP модель
mkdir -p models
wget https://huggingface.co/timm/vit_large_patch14_clip_336.openai/resolve/main/open_clip_model.safetensors \
  -O models/clip-vit-l-14-336.safetensors

# 4. Запустить
uv run python scripts/run_batch.py \
  --clip-checkpoint models/clip-vit-l-14-336.safetensors \
  --device cuda \
  --epochs 5000
```
