import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from rag_structured import (
    load_norms, load_alarms,
    norms_by_equipment, alarms_by_equipment,
    find_norm_by_instrument, find_alarm_by_instrument,
    format_norm_line, format_alarm_line,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

INSTRUMENT_PREFIXES = [
    "PT", "FT", "LT", "TT", "PIC", "FIC", "LIC", "LIT", "PIT",
    "PI", "FI", "TI", "LALL", "FALL", "PDALL",
]

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LMSTUDIO_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
RAG_LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("RAG_LOW_CONFIDENCE_THRESHOLD", "0.35"))
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "900"))
LLM_RETRY_COUNT = int(os.environ.get("LLM_RETRY_COUNT", "2"))

if OPENAI_BASE_URL:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY or "not-needed")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class Chunk:
    doc_name: str
    page: int
    text: str


def _query_intents(query: str) -> List[str]:
    q = (query or "").lower()
    intents = []
    if any(w in q for w in ["устав", "норм", "диапазон", "рабоч", "допустим"]):
        intents.append("norms")
    if any(w in q for w in ["авар", "сигнал", "блокир", "срабатыв", "действие", "останов"]):
        intents.append("alarms")
    if any(w in q for w in ["продукт", "назначен", "раздел", "регламент", "процесс", "состав"]):
        intents.append("reglament")
    return intents


def _guess_instrument_from_query(q: str) -> str:
    prefixes = "|".join(INSTRUMENT_PREFIXES)
    m = re.search(rf"\b((?:{prefixes})-?\d{{3,6}}(?:_\d+)?)\b", (q or "").upper())
    return m.group(1) if m else ""


def _guess_equipment_from_query(q: str) -> str:
    m = re.search(r"\b([FGTD][\s\-–]?\d{3,4}(?:[A-ZА-Я/]{0,3})?)\b", (q or "").upper())
    if not m:
        return ""
    return m.group(1).replace(" ", "").replace("–", "-")


def _extract_instruments(query: str) -> List[str]:
    prefixes = "|".join(INSTRUMENT_PREFIXES)
    found = re.findall(rf"\b((?:{prefixes})-?\d{{3,6}}(?:_\d+)?)\b", (query or "").upper())
    seen, out = set(), []
    for inst in found:
        if inst not in seen:
            seen.add(inst)
            out.append(inst)
    return out


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-я0-9_\-]+", (text or "").lower())


STOPWORDS = {"что", "где", "как", "какие", "какой", "какая", "какое",
             "это", "для", "при", "по", "на", "в", "и", "или", "с", "из", "о"}


def _query_terms(query: str) -> List[str]:
    return [t for t in _tokenize(query) if len(t) >= 3 and t not in STOPWORDS]


def _lex_bm25(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(text)
    if not q_tokens or not d_tokens:
        return 0.0
    tf: Dict[str, int] = {}
    for tok in d_tokens:
        tf[tok] = tf.get(tok, 0) + 1
    k1, b, avgdl = 1.2, 0.75, 160.0
    dl = len(d_tokens)
    score = 0.0
    for tok in set(q_tokens):
        f = tf.get(tok, 0)
        if f == 0:
            continue
        denom = f + k1 * (1 - b + b * dl / avgdl)
        score += f * (k1 + 1) / max(denom, 1e-9)
    return score / max(len(set(q_tokens)), 1)


def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _intent_boost(query: str, doc_name: str) -> float:
    intents = set(_query_intents(query))
    d = (doc_name or "").lower()
    boost = 0.0
    if "norms" in intents and "норм" in d:
        boost += 0.25
    if "alarms" in intents and ("авар" in d or "сигнал" in d):
        boost += 0.25
    if "reglament" in intents and "регламент" in d:
        boost += 0.15
    # штраф за нерелевантный документ
    only_norms = "norms" in intents and "alarms" not in intents
    only_alarms = "alarms" in intents and "norms" not in intents
    if only_norms and ("авар" in d or "сигнал" in d):
        boost -= 0.10
    if only_alarms and "норм" in d:
        boost -= 0.10
    return boost


def _entity_boost(query: str, text: str) -> float:
    bonus = 0.0
    t_up = text.upper()
    inst = _guess_instrument_from_query(query)
    if inst and inst in t_up:
        bonus += 0.18
    eq = _guess_equipment_from_query(query)
    if eq and eq.replace("-", "") in t_up.replace("-", ""):
        bonus += 0.10
    return bonus


def _deduplicate(items: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
    best: Dict[str, Tuple[Chunk, float]] = {}
    for ch, sc in items:
        fp = re.sub(r"\s+", " ", (ch.text or "").strip().lower())
        if fp not in best or sc > best[fp][1]:
            best[fp] = (ch, sc)
    return list(best.values())


def _get_snippet(text: str, max_len: int = 400) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    snip = " ".join(sents[:3]).strip()
    if len(snip) > max_len:
        snip = snip[:max_len] + "..."
    return snip


def _query_focused_snippet(text: str, query: str, max_len: int = 900) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    low = src.lower()
    pos = -1
    for term in _query_terms(query):
        p = low.find(term.lower())
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos == -1:
        return _get_snippet(src, max_len=max_len)
    half = max_len // 2
    start = max(0, pos - half)
    end = min(len(src), pos + half)
    left_dot = src.rfind('.', 0, start)
    right_dot = src.find('.', end)
    if left_dot != -1:
        start = left_dot + 1
    if right_dot != -1:
        end = right_dot + 1
    snippet = src[start:end].strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rstrip() + "..."
    return snippet


class VectorStore:
    def __init__(self, index_path: str, meta_path: str, model_name: Optional[str] = None):
        model_name = model_name or os.environ.get(
            "EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunks = [Chunk(**m) for m in meta]

    def _structured_candidates(self, query: str) -> List[Tuple[Chunk, float]]:
        out: List[Tuple[Chunk, float]] = []
        norms = load_norms()
        alarms = load_alarms()
        for inst in _extract_instruments(query):
            for n in find_norm_by_instrument(norms, inst):
                txt = (f"{n.instrument} {n.param} {n.unit} "
                       f"диапазон {n.range_min}-{n.range_max} "
                       f"рабочий {n.work_min}-{n.work_max}")
                out.append((Chunk("Нормы технологического режима У-300 КТЛ-1.pdf", n.page, txt), 0.98))
            for a in find_alarm_by_instrument(alarms, inst):
                txt = (f"{a.instrument} {a.param} {a.unit} "
                       f"уставка {a.setpoint} действие {a.action} {a.note}")
                out.append((Chunk("Аварии и сигнализации У-300 КТЛ-1.pdf", a.page, txt), 0.98))
        eq = _guess_equipment_from_query(query)
        if eq:
            for n in norms_by_equipment(norms, eq)[:10]:
                txt = f"{n.instrument} {n.param} {n.unit} диапазон {n.range_min}-{n.range_max}"
                out.append((Chunk("Нормы технологического режима У-300 КТЛ-1.pdf", n.page, txt), 0.90))
            for a in alarms_by_equipment(alarms, eq)[:10]:
                txt = f"{a.instrument} {a.param} {a.unit} уставка {a.setpoint} действие {a.action}"
                out.append((Chunk("Аварии и сигнализации У-300 КТЛ-1.pdf", a.page, txt), 0.90))
        return out

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        candidate_k = max(60, k * 8)
        q_emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q_emb, candidate_k)
        out: List[Tuple[Chunk, float]] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            out.append((self.chunks[idx], float(score) + _intent_boost(query, self.chunks[idx].doc_name)))
        out.extend(self._structured_candidates(query))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:candidate_k]


def rerank(query: str, items: List[Tuple[Chunk, float]], top_k: int = 4) -> List[Tuple[Chunk, float]]:
    items = _deduplicate(items)
    if not items:
        return []
    vec = _normalize_scores([s for _, s in items])
    lex = _normalize_scores([fuzz.token_set_ratio(query, ch.text) / 100.0 for ch, _ in items])
    bm25 = _normalize_scores([_lex_bm25(query, ch.text) for ch, _ in items])
    scored: List[Tuple[Chunk, float]] = []
    for i, (ch, _) in enumerate(items):
        s = 0.45 * vec[i] + 0.35 * lex[i] + 0.20 * bm25[i]
        s += _intent_boost(query, ch.doc_name)
        s += _entity_boost(query, ch.text)
        # короткие/пустые фрагменты считаем шумом
        if len((ch.text or "").strip()) < 40:
            s -= 0.25
        scored.append((ch, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def validate_llm_config() -> Tuple[bool, str]:
    if OPENAI_BASE_URL:
        if OPENAI_API_KEY:
            return True, "ok"
        if any(h in OPENAI_BASE_URL for h in ("localhost", "127.0.0.1", "host.docker.internal")):
            return True, "ok"
        return False, "Set OPENAI_API_KEY for non-local OPENAI_BASE_URL"
    if not OPENAI_API_KEY:
        return False, "Set OPENAI_API_KEY"
    return True, "ok"


def _fallback_answer(query: str, ctx: List[Chunk]) -> Dict:
    citations = [{"doc_name": ch.doc_name, "page": ch.page,
                  "snippet": _get_snippet(ch.text, max_len=300)} for ch in ctx[:3]]
    return {
        "answer": ("Недостаточно уверенности в найденном контексте. "
                   "Уточните вопрос: добавьте номер прибора, оборудования или параметра."),
        "citations": citations,
    }


SYSTEM_PROMPT = """Ты инженер-консультант по установке У-300 КТЛ-1.
Пиши только на русском языке и только по данным из предоставленного контекста.

Правила:
1) Не выдумывай факты, значения, теги, страницы.
2) Если данных недостаточно, явно так и напиши.
3) Для вопросов по оборудованию: краткий вывод, найденные нормы,
   найденные сигнализации, цитаты (документ и страница).
4) Для вопросов по прибору: нормы/уставки и сигнализации по прибору,
   связанное оборудование, если определяется. Если данных нет — скажи об этом.
5) Для общих вопросов: краткий вывод, пояснение по процессу, цитаты.
"""


def _build_context(query: str, ctx: List[Chunk]) -> str:
    blocks = [f"{ch.doc_name}, стр. {ch.page}: {_query_focused_snippet(ch.text, query)}"
              for ch in ctx]
    text = "\n\n".join(blocks)
    instrument = _guess_instrument_from_query(query)
    equipment = _guess_equipment_from_query(query)
    norms = load_norms()
    alarms = load_alarms()
    norm_lines: List[str] = []
    alarm_lines: List[str] = []
    if instrument:
        inst_norms = find_norm_by_instrument(norms, instrument)
        inst_alarms = find_alarm_by_instrument(alarms, instrument)
        norm_lines.extend(format_norm_line(n) for n in inst_norms)
        alarm_lines.extend(filter(None, (format_alarm_line(a) for a in inst_alarms)))
        if inst_norms and not equipment:
            equipment = inst_norms[0].equipment
        elif inst_alarms and not equipment:
            equipment = inst_alarms[0].equipment
    if equipment and not instrument:
        norm_lines.extend(format_norm_line(n) for n in norms_by_equipment(norms, equipment))
        alarm_lines.extend(filter(None, (format_alarm_line(a) for a in alarms_by_equipment(alarms, equipment))))
    if norm_lines:
        text += "\n\n[Нормальные значения]:\n" + "\n\n".join(norm_lines)
    if alarm_lines:
        text += "\n\n[Сигнализации]:\n" + "\n\n".join(alarm_lines)
    return text


def make_answer_llm(query: str, ctx: List[Chunk], model: str = DEFAULT_LLM_MODEL,
                    retrieval_score: Optional[float] = None) -> Dict:
    ok, msg = validate_llm_config()
    if not ok:
        raise RuntimeError(f"LLM config error: {msg}")
    if not ctx or (retrieval_score is not None and retrieval_score < RAG_LOW_CONFIDENCE_THRESHOLD):
        return _fallback_answer(query, ctx)

    context = _build_context(query, ctx)
    user_prompt = f"Вопрос: {query}\n\nКонтекст:\n{context}"

    start = time.perf_counter()
    resp = None
    for attempt in range(LLM_RETRY_COUNT + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            break
        except Exception as exc:
            if attempt >= LLM_RETRY_COUNT:
                raise
            logger.warning("LLM попытка %s упала: %s", attempt + 1, exc)
            time.sleep(0.8 * (attempt + 1))

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    answer = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    if usage:
        logger.info("LLM model=%s latency_ms=%s prompt=%s completion=%s",
                    model, elapsed_ms,
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None))
    else:
        logger.info("LLM model=%s latency_ms=%s", model, elapsed_ms)

    citations = [{"doc_name": ch.doc_name, "page": ch.page,
                  "snippet": _get_snippet(ch.text, max_len=300)} for ch in ctx]
    return {"answer": answer, "citations": citations, "latency_ms": elapsed_ms}
