FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -c "import nltk; import ssl; ssl._create_default_https_context=ssl._create_unverified_context; import nltk; nltk.download('punkt', quiet=True)" || true

# Allow configuring the LM Studio / OpenAI base URL from environment
ENV LMSTUDIO_URL="http://host.docker.internal:1234/v1"

# Копируем проект
COPY . .

# Порт для Streamlit
EXPOSE 8501

# Запуск Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
