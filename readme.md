# RAG-U300-KTL1

Проект Retrieval-Augmented Generation (RAG) на документации установки **У-300 КТЛ-1**.  
Корпус: три PDF из эксплуатационной документации:
- Аварии и сигнализации У-300 КТЛ-1.pdf
- Нормы технологического режима У-300 КТЛ-1.pdf
- Технологический регламент У-300 КТЛ-1.pdf

Цель: отвечать на вопросы строго по документации, с точными цитатами «Документ → страница».

---

## 🚀 Быстрый старт

```bash
# 1. Создаём виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Устанавливаем зависимости
pip install -r requirements.txt

# 3. Индексация PDF → FAISS
python ingest.py --data_dir data --out_dir storage

# 4. Извлечение таблиц (уставки, теги LIC/PDT/PDI/FT)
python table_extractor.py --data_dir data --out storage/tables_kv.json

# 5. Запуск Streamlit интерфейса
streamlit run app.py
