from dataclasses import dataclass
from typing import List, Dict, Tuple
import json, os, re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz import fuzz
from openai import OpenAI

from rag_structured import (
    load_norms, load_alarms,
    norms_by_equipment, alarms_by_equipment,
    find_norm_by_instrument, find_alarm_by_instrument,
    format_norm_line, format_alarm_line
)

# ------------------------------
# Базовые структуры
# ------------------------------
@dataclass
class Chunk:
    doc_name: str
    page: int
    text: str

class VectorStore:
    def __init__(self, index_path: str, meta_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        self.chunks = [Chunk(**m) for m in meta]

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q, k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1: continue
            out.append((self.chunks[idx], float(score)))
        return out

# ------------------------------
# Rerank
# ------------------------------
def rerank(query: str, items: List[Tuple[Chunk, float]], top_k: int = 4) -> List[Tuple[Chunk, float]]:
    scored = []
    for ch, vec_score in items:
        lex = fuzz.token_set_ratio(query, ch.text) / 100.0
        score = 0.5 * vec_score + 0.5 * lex
        scored.append((ch, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# ------------------------------
# LM Studio client
# ------------------------------
# Allow overriding LM Studio / OpenAI base URL via environment variable so container
# can point to host.docker.internal or a different endpoint.
LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
client = OpenAI(base_url=LMSTUDIO_URL, api_key=OPENAI_API_KEY)

def _get_snippet(text: str, max_len: int = 400) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    snip = " ".join(sents[:3]).strip()
    if len(snip) > max_len:
        snip = snip[:max_len] + "..."
    return snip

def _normalize_equipment_name(name: str) -> str:
    if not name:
        return ""
    name = name.upper().replace(" ", "").replace("–", "-")
    name = name.replace("А", "A").replace("В", "B")
    return name

def _guess_equipment_from_query(q: str) -> str:
    m = re.search(r"\b([FGTD][\s\-–]?\d{3,4}(?:[A-ZА-Я/]{0,3})?)\b", q.upper())
    if not m:
        return ""
    raw = m.group(1)
    return _normalize_equipment_name(raw)

def _guess_instrument_from_query(q: str) -> str:
    m = re.search(r"\b([A-Z]{2,3}-\d{4,6})\b", q.upper())
    return m.group(1) if m else ""

# ------------------------------
# Основная сборка ответа
# ------------------------------
def make_answer_llm(query: str, ctx: List[Chunk], model: str = "meta-llama-3-8b-instruct-bf16-correct-pre-tokenizer-and-eos-token-q8_0-q6_k") -> Dict:
    # 1) Векторный контекст
    context_blocks = [ f"{ch.doc_name}, стр. {ch.page}: {_get_snippet(ch.text)}" for ch in ctx ]
    context = "\n\n".join(context_blocks)

    # 2) Структурные данные
    equipment = _guess_equipment_from_query(query)
    instrument = _guess_instrument_from_query(query)

    norms = load_norms()
    alarms = load_alarms()

    norm_lines, alarm_lines = [], []

    if instrument:
        # поиск по прибору
        inst_norms = find_norm_by_instrument(norms, instrument)
        inst_alarms = find_alarm_by_instrument(alarms, instrument)

        if inst_norms:
            norm_lines.extend([format_norm_line(n) for n in inst_norms])
            equipment = inst_norms[0].equipment
        if inst_alarms:
            alarm_lines.extend([format_alarm_line(a) for a in inst_alarms if format_alarm_line(a)])
            if not equipment and inst_alarms:
                equipment = inst_alarms[0].equipment

    if equipment and not instrument:
        for n in norms_by_equipment(norms, equipment):
            norm_lines.append(format_norm_line(n))
        for a in alarms_by_equipment(alarms, equipment):
            line = format_alarm_line(a)
            if line:
                alarm_lines.append(line)
    if norm_lines:
        context += "\n\n[Нормальные значения]:\n" + "\n\n".join(norm_lines)
    if alarm_lines:
        context += "\n\n[Сигнализации]:\n" + "\n\n".join(alarm_lines)
        
        
    # 3) Промпт
    prompt = f"""
Ты инженер-консультант по установке У-300 КТЛ-1.
Отвечай строго по документации и **всегда только на русском языке**,
даже если встречаются англоязычные термины в таблицах или описаниях.

Если вопрос касается конкретного оборудования (например, "G-303", "G-304", "D-301", "F-301" и т.п.), то:
1) Краткий вывод (1–2 предложения).
2) Подробное пояснение (назначение, работа, роль в процессе, минимум 6 предложений).
3) Нормальные значения (создай таблицу с полным списком нормальных значений и объясни, что означают рабочие и допустимые диапазоны).
4) Сигнализации (Сигнализации: перечисли абсолютно все найденные сигнализации для данного оборудования.
Составь таблицу с колонками:
- № сигнализации
- Параметр
- Уставка
- Действие при срабатывании
- Примечание (если есть)
5) Цитаты (Документ / страница).

Если вопрос будет про прибор название которых начинается с "PT, FT, LT, TT, PIC, FIC, LIC, LIT, PIT, PI, FI, TI, LALL, FALL, PDALL", то: 
1) обязательно приводи нормальные значения и все сигнализации именно по этому прибору. 
2) Напиши к какому оборудованию этот прибор относится(если это возможно определить из названия прибора, например этот прибор относится к g-303, t-301? f-305, d-301).
3) Если прибор не найден в структурированных данных, то ответь "По прибору {{название прибора}} в документации нет данных по нормальным значениям и аварийным сигнализациям."

Если вопрос общий, без упоминания конкретного оборудования, то:
1) Краткий вывод (1–2 предложения).
2) Подробное пояснение (назначение, работа, роль в процессе, минимум 6 предложений).
3) Цитаты (Документ / страница).


Не выдумывай. Пиши техническим стилем, но **целиком на русском языке**.

Вопрос: {query}

Контент:
{context}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35
    )
    answer = resp.choices[0].message.content

    # 4) Цитаты
    citations = []
    for ch in ctx:
        citations.append({
            "doc_name": ch.doc_name,
            "page": ch.page,
            "snippet": _get_snippet(ch.text, max_len=300)
        })
    return {"answer": answer, "citations": citations}
