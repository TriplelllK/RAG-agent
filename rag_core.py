from dataclasses import dataclass
from typing import List, Dict, Tuple
import json, os, re, math, logging, time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz import fuzz
from openai import OpenAI
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

from rag_structured import (
    load_norms, load_alarms,
    norms_by_equipment, alarms_by_equipment,
    find_norm_by_instrument, find_alarm_by_instrument,
    format_norm_line, format_alarm_line
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

def _load_api_key_from_file() -> str:
    key_file = os.path.join(os.path.dirname(__file__), "openai_api_key.txt")
    if not os.path.exists(key_file):
        return ""
    try:
        with open(key_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        logger.warning("Could not read openai_api_key.txt")
        return ""

def _query_intents(query: str) -> List[str]:
    q = (query or "").lower()
    intents = []
    if any(w in q for w in ["устав", "норм", "диапазон", "рабоч", "допустим"]):
        intents.append("norms")
    if any(w in q for w in ["авар", "сигнал", "блокир", "срабатыв", "действие", "останов", "закрыти", "открыти"]):
        intents.append("alarms")
    if any(w in q for w in ["продукт", "назначен", "раздел", "регламент", "технологич", "процесс", "состав"]):
        intents.append("reglament")
    return intents

def _intent_expansion_queries(query: str) -> List[str]:
    intents = _query_intents(query)
    expanded = [query]
    if "norms" in intents:
        expanded.append(f"{query} нормы технологического режима уставка допустимый рабочий диапазон")
    if "alarms" in intents:
        expanded.append(f"{query} аварии и сигнализации блокировки действия при срабатывании")
    if "reglament" in intents:
        expanded.append(f"{query} технологический регламент раздел")
    return expanded

def _extract_instruments(query: str) -> List[str]:
    prefixes = "|".join(INSTRUMENT_PREFIXES)
    found = re.findall(rf"\b((?:{prefixes})-?\d{{3,6}}(?:_\d+)?)\b", (query or "").upper())
    uniq = []
    seen = set()
    for inst in found:
        if inst not in seen:
            uniq.append(inst)
            seen.add(inst)
    return uniq

# Простые структуры
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

        self._doc_counts: Dict[str, int] = {}
        for ch in self.chunks:
            self._doc_counts[ch.doc_name] = self._doc_counts.get(ch.doc_name, 0) + 1

        total = max(len(self.chunks), 1)
        self._doc_prior: Dict[str, float] = {}
        raw_vals = []
        for doc, count in self._doc_counts.items():
            val = math.log((total + 1) / (count + 1))
            self._doc_prior[doc] = val
            raw_vals.append(val)

        lo = min(raw_vals) if raw_vals else 0.0
        hi = max(raw_vals) if raw_vals else 1.0
        denom = max(hi - lo, 1e-9)
        for doc in list(self._doc_prior.keys()):
            self._doc_prior[doc] = (self._doc_prior[doc] - lo) / denom

    def _doc_prior_boost(self, query: str, doc_name: str) -> float:
        intents = _query_intents(query)
        boost = 0.0
        dn = (doc_name or "").lower()

        if "norms" in intents and "норм" in dn:
            boost += 0.20
        if "alarms" in intents and ("авар" in dn or "сигнал" in dn):
            boost += 0.20
        if "reglament" in intents and "регламент" in dn:
            boost += 0.12

        inst = _guess_instrument_from_query(query)
        if inst and ("норм" in dn or "авар" in dn or "сигнал" in dn):
            boost += 0.12

        boost += 0.10 * self._doc_prior.get(doc_name, 0.0)
        return min(boost, 0.35)

    def _structured_candidates(self, query: str) -> List[Tuple[Chunk, float]]:
        candidates: List[Tuple[Chunk, float]] = []

        instruments = _extract_instruments(query)
        equipment = _guess_equipment_from_query(query)

        norms = load_norms()
        alarms = load_alarms()

        for inst in instruments:
            for n in find_norm_by_instrument(norms, inst):
                txt = f"{n.instrument} {n.param} {n.unit} диапазон {n.range_min}-{n.range_max} рабочий {n.work_min}-{n.work_max}"
                candidates.append((Chunk(doc_name="Нормы технологического режима У-300 КТЛ-1.pdf", page=n.page, text=txt), 0.98))
            for a in find_alarm_by_instrument(alarms, inst):
                txt = f"{a.instrument} {a.param} {a.unit} уставка {a.setpoint} действие {a.action} {a.note}"
                candidates.append((Chunk(doc_name="Аварии и сигнализации У-300 КТЛ-1.pdf", page=a.page, text=txt), 0.98))

        if equipment:
            for n in norms_by_equipment(norms, equipment)[:10]:
                txt = f"{n.instrument} {n.param} {n.unit} диапазон {n.range_min}-{n.range_max}"
                candidates.append((Chunk(doc_name="Нормы технологического режима У-300 КТЛ-1.pdf", page=n.page, text=txt), 0.90))
            for a in alarms_by_equipment(alarms, equipment)[:10]:
                txt = f"{a.instrument} {a.param} {a.unit} уставка {a.setpoint} действие {a.action}"
                candidates.append((Chunk(doc_name="Аварии и сигнализации У-300 КТЛ-1.pdf", page=a.page, text=txt), 0.90))

        return candidates

    def _keyword_scan_candidates(self, query: str, limit: int = 20) -> List[Tuple[Chunk, float]]:
        intents = set(_query_intents(query))
        if "reglament" not in intents:
            return []

        terms = _query_terms(query)
        if not terms:
            return []

        scored: List[Tuple[Chunk, float]] = []
        for ch in self.chunks:
            dn = (ch.doc_name or "").lower()
            if "регламент" not in dn:
                continue

            txt = (ch.text or "")
            low = txt.lower()
            hits = 0
            for t in terms[:8]:
                if t in low:
                    hits += 1

            if hits == 0:
                continue

            # Легкий бонус для "продуктовых" запросов.
            bonus = 0.0
            q = (query or "").lower()
            if "продукт" in q and any(w in low for w in ["продукт", "очищ", "кисл", "газ"]):
                bonus += 0.20

            score = min(0.55 + 0.08 * hits + bonus, 0.96)
            scored.append((ch, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        candidate_k = max(60, k * 8)
        score_by_idx: Dict[int, float] = {}

        for expanded_query in _intent_expansion_queries(query):
            q = self.model.encode([expanded_query], normalize_embeddings=True)
            D, I = self.index.search(q, candidate_k)
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx == -1:
                    continue
                base = float(score)
                boosted = base + self._doc_prior_boost(query, self.chunks[idx].doc_name)
                score_by_idx[idx] = max(boosted, score_by_idx.get(idx, -1.0))

        out = [(self.chunks[idx], sc) for idx, sc in score_by_idx.items()]
        out.extend(self._structured_candidates(query))
        out.extend(self._keyword_scan_candidates(query, limit=max(12, k * 2)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:candidate_k]

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-я0-9_\-]+", (text or "").lower())

def _query_terms(query: str) -> List[str]:
    stop = {
        "что", "где", "как", "какие", "какой", "какая", "какое", "это", "для", "при", "по", "на", "в", "и", "или", "с", "из", "о"
    }
    terms = []
    for t in _tokenize(query):
        if len(t) < 3:
            continue
        if t in stop:
            continue
        terms.append(t)
    return terms

def _bm25_like_score(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    d_tokens = _tokenize(text)
    if not q_tokens or not d_tokens:
        return 0.0

    tf = {}
    for tok in d_tokens:
        tf[tok] = tf.get(tok, 0) + 1

    score = 0.0
    dl = len(d_tokens)
    avgdl = 160.0
    k1, b = 1.2, 0.75
    for tok in set(q_tokens):
        f = tf.get(tok, 0)
        if f == 0:
            continue
        idf = 1.0
        denom = f + k1 * (1 - b + b * dl / avgdl)
        score += idf * (f * (k1 + 1)) / max(denom, 1e-9)

    return score / max(len(set(q_tokens)), 1)

def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def _text_fingerprint(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return t

def _is_boilerplate_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if len(t) < 40:
        return True
    if "технологический регламент установки 300 ктл-1" in t and len(t) < 90:
        return True
    tokens = _tokenize(t)
    if len(set(tokens)) <= 5 and len(t) < 120:
        return True
    return False

def _boilerplate_penalty(text: str) -> float:
    return -0.28 if _is_boilerplate_text(text) else 0.0

def _deduplicate_items(items: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
    if not items:
        return []
    best_by_fp: Dict[str, Tuple[Chunk, float]] = {}
    for ch, vec in items:
        fp = _text_fingerprint(ch.text)
        prev = best_by_fp.get(fp)
        if prev is None or vec > prev[1]:
            best_by_fp[fp] = (ch, vec)
    return list(best_by_fp.values())

def _doc_intent_boost(query: str, doc_name: str) -> float:
    q = query.lower()
    d = (doc_name or "").lower()
    boost = 0.0

    if any(w in q for w in ["устав", "норм", "диапазон", "рабоч"]):
        if "норм" in d:
            boost += 0.25

    if any(w in q for w in ["авар", "сигнал", "блокир", "действие", "срабатыв"]):
        if "авар" in d or "сигнал" in d:
            boost += 0.25

    if any(w in q for w in ["продукт", "назначен", "описан", "раздел", "состав", "процесс"]):
        if "регламент" in d:
            boost += 0.18

    return min(boost, 0.35)

def _doc_intent_penalty(query: str, doc_name: str) -> float:
    intents = set(_query_intents(query))
    if not intents:
        return 0.0

    d = (doc_name or "").lower()
    is_norm_doc = "норм" in d
    is_alarm_doc = "авар" in d or "сигнал" in d

    only_norms = "norms" in intents and "alarms" not in intents
    only_alarms = "alarms" in intents and "norms" not in intents

    penalty = 0.0
    if only_norms and is_alarm_doc:
        penalty -= 0.10
    if only_alarms and is_norm_doc:
        penalty -= 0.10
    return penalty

def _doc_quota_for_query(query: str, top_k: int) -> int:
    intents = set(_query_intents(query))
    if len(intents) >= 2:
        return max(1, top_k // 2)
    if "reglament" in intents and len(intents) == 1:
        return max(1, top_k)
    return max(1, top_k // 2)

def _apply_doc_quota(scored: List[Tuple[Chunk, float]], query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    if not scored:
        return []

    max_per_doc = _doc_quota_for_query(query, top_k)
    picked: List[Tuple[Chunk, float]] = []
    per_doc: Dict[str, int] = {}

    for ch, sc in scored:
        cnt = per_doc.get(ch.doc_name, 0)
        if cnt >= max_per_doc:
            continue
        picked.append((ch, sc))
        per_doc[ch.doc_name] = cnt + 1
        if len(picked) >= top_k:
            return picked

    if len(picked) < top_k:
        for ch, sc in scored:
            if len(picked) >= top_k:
                break
            if any((x.doc_name == ch.doc_name and x.page == ch.page and x.text == ch.text) for x, _ in picked):
                continue
            picked.append((ch, sc))

    return picked

def _exact_entity_boost(query: str, text: str) -> float:
    q_up = query.upper()
    t_up = text.upper()
    bonus = 0.0

    inst = _guess_instrument_from_query(query)
    if inst and inst in t_up:
        bonus += 0.18

    eq = _guess_equipment_from_query(query)
    if eq and eq.replace("-", "") in t_up.replace("-", ""):
        bonus += 0.10

    if any(token in q_up for token in ["УСТАВ", "НОРМ"]) and any(token in t_up for token in ["ДОПУСТИМ", "РАБОЧИЙ", "ДИАПАЗОН"]):
        bonus += 0.05

    return min(bonus, 0.25)

def _text_question_boost(query: str, text: str, doc_name: str) -> float:
    intents = set(_query_intents(query))
    if "reglament" not in intents:
        return 0.0

    d = (doc_name or "").lower()
    if "регламент" not in d:
        return 0.0

    t = (text or "").lower()
    q = (query or "").lower()
    boost = 0.0

    if "продукт" in q:
        if any(w in t for w in ["продукт", "очищ", "кисл", "газ"]):
            boost += 0.12

    if any(w in q for w in ["состав", "описан", "назначен", "раздел"]):
        if any(w in t for w in ["состав", "раздел", "опис", "назнач"]):
            boost += 0.08

    if any(term in t for term in _query_terms(query)[:6]):
        boost += 0.05

    return min(boost, 0.18)

def rerank(query: str, items: List[Tuple[Chunk, float]], top_k: int = 4) -> List[Tuple[Chunk, float]]:
    items = _deduplicate_items(items)
    if not items:
        return []

    vec_raw = [vec for _, vec in items]
    lex_raw = [fuzz.token_set_ratio(query, ch.text) / 100.0 for ch, _ in items]
    bm25_raw = [_bm25_like_score(query, ch.text) for ch, _ in items]

    vec_norm = _normalize_scores(vec_raw)
    lex_norm = _normalize_scores(lex_raw)
    bm25_norm = _normalize_scores(bm25_raw)

    scored = []
    for idx, (ch, _) in enumerate(items):
        score = 0.45 * vec_norm[idx] + 0.35 * lex_norm[idx] + 0.20 * bm25_norm[idx]
        score += _doc_intent_boost(query, ch.doc_name)
        score += _doc_intent_penalty(query, ch.doc_name)
        score += _exact_entity_boost(query, ch.text)
        score += _text_question_boost(query, ch.text, ch.doc_name)
        score += _boilerplate_penalty(ch.text)
        scored.append((ch, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return _apply_doc_quota(scored, query=query, top_k=top_k)

# Если задан OPENAI_BASE_URL или LMSTUDIO_URL, используем его.
# Иначе идет запрос в стандартный OpenAI endpoint.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LMSTUDIO_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or _load_api_key_from_file()
DEFAULT_LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
RAG_LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("RAG_LOW_CONFIDENCE_THRESHOLD", "0.35"))
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "900"))
LLM_RETRY_COUNT = int(os.environ.get("LLM_RETRY_COUNT", "2"))

INSTRUMENT_PREFIXES = [
    "PT", "FT", "LT", "TT", "PIC", "FIC", "LIC", "LIT", "PIT", "PI", "FI", "TI", "LALL", "FALL", "PDALL",
]

EQUIPMENT_SYNONYMS = {
    "АБСОРБЕР": "D",
    "СЕПАРАТОР": "F",
    "НАСОС": "G",
    "КОЛОННА": "D",
    "ФИЛЬТР": "F",
    "СКРУББЕР": "F",
    "РЕГЕНЕРАТОР": "D",
    "КОНТАКТОР": "D",
}

if OPENAI_BASE_URL:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY or "not-needed")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

def validate_llm_config() -> Tuple[bool, str]:
    if OPENAI_BASE_URL:
        if OPENAI_API_KEY:
            return True, "ok"
        if "localhost" in OPENAI_BASE_URL or "127.0.0.1" in OPENAI_BASE_URL or "host.docker.internal" in OPENAI_BASE_URL:
            return True, "ok"
        return False, "Set OPENAI_API_KEY for non-local OPENAI_BASE_URL"

    if not OPENAI_API_KEY:
        return False, "Set OPENAI_API_KEY"
    return True, "ok"

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
    terms = _query_terms(query)
    pos = -1
    best_term = ""
    for term in terms:
        p = low.find(term.lower())
        if p != -1 and (pos == -1 or p < pos):
            pos = p
            best_term = term

    if pos == -1:
        return _get_snippet(src, max_len=max_len)

    half = max_len // 2
    start = max(0, pos - half)
    end = min(len(src), pos + half)

    # Попробуем расширить до границ предложений
    left_dot = src.rfind('.', 0, start)
    right_dot = src.find('.', end)
    if left_dot != -1:
        start = left_dot + 1
    if right_dot != -1:
        end = right_dot + 1

    snippet = src[start:end].strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rstrip() + "..."

    if len(snippet) < 120 and best_term:
        return _get_snippet(src, max_len=max_len)
    return snippet

def _normalize_equipment_name(name: str) -> str:
    if not name:
        return ""
    name = name.upper().replace(" ", "").replace("–", "-")
    name = name.replace("А", "A").replace("В", "B")
    return name

def _replace_equipment_words(q: str) -> str:
    up = q.upper()
    for word, code in EQUIPMENT_SYNONYMS.items():
        up = up.replace(word, code)
    return up

def _guess_equipment_from_query(q: str) -> str:
    src = _replace_equipment_words(q)
    m = re.search(r"\b([FGTD][\s\-–]?\d{3,4}(?:[A-ZА-Я/]{0,3})?)\b", src.upper())
    if not m:
        return ""
    raw = m.group(1)
    return _normalize_equipment_name(raw)

def _guess_instrument_from_query(q: str) -> str:
    prefixes = "|".join(INSTRUMENT_PREFIXES)
    m = re.search(rf"\b((?:{prefixes})-?\d{{3,6}}(?:_\d+)?)\b", q.upper())
    return m.group(1) if m else ""

def _fallback_answer(query: str, ctx: List[Chunk]) -> Dict:
    citations = []
    for ch in ctx[:3]:
        citations.append({
            "doc_name": ch.doc_name,
            "page": ch.page,
            "snippet": _get_snippet(ch.text, max_len=300)
        })
    return {
        "answer": "Недостаточно уверенности в найденном контексте. Уточните вопрос: добавьте номер прибора, оборудования или параметра.",
        "citations": citations
    }

def _try_answer_products_question(query: str) -> Dict | None:
    q = (query or "").lower()
    if "продукт" not in q or "установ" not in q:
        return None

    meta_path = os.path.join(os.path.dirname(__file__), "storage", "meta.json")
    if not os.path.exists(meta_path):
        return None

    try:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
    except Exception:
        return None

    candidates = []
    for row in meta:
        doc = (row.get("doc_name") or "")
        if "регламент" not in doc.lower():
            continue
        txt = (row.get("text") or "")
        low = txt.lower()
        if "продуктами установки 300" in low or "продуктами установки" in low:
            candidates.append(row)

    if not candidates:
        return None

    best = candidates[0]
    text = (best.get("text") or "")
    lines = [x.strip(" •;\t") for x in re.split(r"\n+", text) if x.strip()]

    products = []
    for ln in lines:
        l = ln.lower()
        if any(k in l for k in ["очищенный газ", "кислый газ", "конденсат", "установку 700", "установку 200", "у-400"]):
            products.append(ln.rstrip(".;"))

    # Если не удалось выделить маркеры, используем блок после двоеточия.
    if not products and ":" in text:
        tail = text.split(":", 1)[1]
        parts = [p.strip(" •;\n\t") for p in re.split(r"[\n;]+", tail) if p.strip()]
        for p in parts[:8]:
            if len(p) > 6:
                products.append(p.rstrip("."))

    if not products:
        return None

    answer_lines = ["Краткий вывод:", "Продуктами установки 300 являются:"]
    for p in products[:6]:
        answer_lines.append(f"- {p}")
    answer_lines.append("")
    answer_lines.append("Цитата: данные взяты из технологического регламента.")

    citation = {
        "doc_name": best.get("doc_name", "Технологический регламент У-300 КТЛ-1.pdf"),
        "page": int(best.get("page") or 1),
        "snippet": _get_snippet(text, max_len=350)
    }
    return {"answer": "\n".join(answer_lines), "citations": [citation]}

def make_answer_llm(query: str, ctx: List[Chunk], model: str = DEFAULT_LLM_MODEL, retrieval_score: float | None = None) -> Dict:
    cfg_ok, cfg_msg = validate_llm_config()
    if not cfg_ok:
        raise RuntimeError(f"LLM config error: {cfg_msg}")

    if retrieval_score is not None and retrieval_score < RAG_LOW_CONFIDENCE_THRESHOLD:
        return _fallback_answer(query, ctx)

    if not ctx:
        return _fallback_answer(query, ctx)

    extracted = _try_answer_products_question(query)
    if extracted is not None:
        return extracted

    # 1) Контекст из вектора
    context_blocks = [ f"{ch.doc_name}, стр. {ch.page}: {_query_focused_snippet(ch.text, query)}" for ch in ctx ]
    context = "\n\n".join(context_blocks)

    # 2) Структурные данные
    equipment = _guess_equipment_from_query(query)
    instrument = _guess_instrument_from_query(query)

    norms = load_norms()
    alarms = load_alarms()

    norm_lines, alarm_lines = [], []

    if instrument:
        # Поиск по прибору
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
    system_prompt = """
Ты инженер-консультант по установке У-300 КТЛ-1.
Пиши только на русском языке и только по данным из предоставленного контекста.

Правила:
1) Не выдумывай факты, значения, теги, страницы.
2) Если данных недостаточно, явно так и напиши.
3) Для вопросов по оборудованию:
   - краткий вывод,
   - сжатое техническое пояснение,
   - найденные нормы,
   - найденные сигнализации,
   - цитаты (документ и страница).
4) Для вопросов по прибору:
   - приведи нормы/уставки и сигнализации именно по прибору,
   - укажи связанное оборудование, если определяется,
   - если данных нет, явно сообщи об этом.
5) Для общих вопросов:
   - краткий вывод,
   - пояснение по процессу,
   - цитаты.

Форматируй ответ структурно, но без избыточной длины.
""".strip()

    user_prompt = f"""
Вопрос: {query}

Контекст:
{context}
""".strip()

    start = time.perf_counter()
    last_error = None
    for attempt in range(LLM_RETRY_COUNT + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_RETRY_COUNT:
                raise
            time.sleep(0.8 * (attempt + 1))

    if last_error and 'resp' not in locals():
        raise RuntimeError(f"LLM call failed: {last_error}")

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    answer = resp.choices[0].message.content

    usage = getattr(resp, "usage", None)
    if usage:
        in_toks = getattr(usage, "prompt_tokens", None)
        out_toks = getattr(usage, "completion_tokens", None)
        total_toks = getattr(usage, "total_tokens", None)
        logger.info("LLM call model=%s latency_ms=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s", model, elapsed_ms, in_toks, out_toks, total_toks)
    else:
        logger.info("LLM call model=%s latency_ms=%s", model, elapsed_ms)

    # 4) Цитаты
    citations = []
    for ch in ctx:
        citations.append({
            "doc_name": ch.doc_name,
            "page": ch.page,
            "snippet": _get_snippet(ch.text, max_len=300)
        })
    return {"answer": answer, "citations": citations, "latency_ms": elapsed_ms}
