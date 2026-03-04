import json, argparse
import os, sys
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag_core import VectorStore, rerank, make_answer_llm
from rag_structured import load_norms, load_alarms, find_norm_by_instrument, find_alarm_by_instrument

parser = argparse.ArgumentParser()
parser.add_argument('--gold', default='eval/gold_qa.jsonl')
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--search_k', type=int, default=10)
parser.add_argument('--no-llm', action='store_true')
args = parser.parse_args()

store = VectorStore('storage/faiss.index', 'storage/meta.json')
norms = load_norms()
alarms = load_alarms()

recall_hits = 0
faithful_hits = 0
citation_hits = 0
instrument_queries = 0
instrument_hits = 0
mrr_sum = 0.0
llm_errors = 0
n = 0

def _guess_instrument(q: str) -> str:
    m = re.search(r"\b([A-Z]{2,5}-?\d{3,6}(?:_\d+)?)\b", q.upper())
    return m.group(1) if m else ""

for line in open(args.gold, 'r', encoding='utf-8'):
    n += 1
    ex = json.loads(line)

    raw = store.search(ex['q'], k=args.search_k)
    top = rerank(ex['q'], raw, top_k=args.k)

    ctx = [x[0] for x in top]
    top_score = top[0][1] if top else 0.0

    # Проверяем, нашли ли нужный документ.
    if any(ex['doc'] in ch.doc_name for ch, _ in top):
        recall_hits += 1

    rank = None
    for idx, (ch, _) in enumerate(top, start=1):
        if ex['doc'] in ch.doc_name:
            rank = idx
            break
    if rank is not None:
        mrr_sum += 1.0 / rank

    inst = _guess_instrument(ex['q'])
    if inst:
        instrument_queries += 1
        if find_norm_by_instrument(norms, inst) or find_alarm_by_instrument(alarms, inst):
            instrument_hits += 1

    # Проверяем, есть ли ключи в цитатах.
    if args.no_llm:
        citations = [{"snippet": ch.text[:300]} for ch in ctx]
    else:
        try:
            out = make_answer_llm(ex['q'], ctx, retrieval_score=top_score)
            citations = out.get('citations', [])
        except Exception:
            llm_errors += 1
            citations = [{"snippet": ch.text[:300]} for ch in ctx]

    if citations:
        citation_hits += 1

    cit_blob = "\n".join([c.get('snippet', '') for c in citations])
    tokens = ex.get('a_contains', [])
    if tokens and all(tok.lower() in cit_blob.lower() for tok in tokens):
        faithful_hits += 1

print({
    "N": n,
    f"Recall@{args.k}(doc-hint)": (recall_hits / n) if n else 0.0,
    f"MRR@{args.k}(doc-hint)": (mrr_sum / n) if n else 0.0,
    "Faithfulness(proxy)": (faithful_hits / n) if n else 0.0,
    "Answer with citation rate": (citation_hits / n) if n else 0.0,
    "Instrument hit rate": (instrument_hits / instrument_queries) if instrument_queries else 0.0,
    "LLM errors": llm_errors,
    "No LLM mode": args.no_llm,
})
