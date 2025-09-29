import json, argparse
from rag_core import VectorStore, rerank, rerank_ce, make_answer

parser = argparse.ArgumentParser()
parser.add_argument('--gold', default='eval/gold_qa.jsonl')
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--rerank', choices=['rapidfuzz', 'cross-encoder'], default='rapidfuzz')
args = parser.parse_args()

store = VectorStore('storage/faiss.index', 'storage/meta.json')

recall_hits = 0
faithful_hits = 0
n = 0

for line in open(args.gold, 'r', encoding='utf-8'):
    n += 1
    ex = json.loads(line)

    raw = store.search(ex['q'], k=10)
    if args.rerank == 'cross-encoder':
        top = rerank_ce(ex['q'], raw, top_k=args.k)
    else:
        top = rerank(ex['q'], raw, top_k=args.k)

    ctx = [x[0] for x in top]

    # Recall@k proxy: нашли ли мы правильный документ
    if any(ex['doc'] in ch.doc_name for ch, _ in top):
        recall_hits += 1

    # Faithfulness proxy: содержатся ли ожидаемые ключи в цитатах
    out = make_answer(ex['q'], ctx)
    cit_blob = "\n".join([c['snippet'] for c in out['citations']])
    if all(tok.lower() in cit_blob.lower() for tok in ex.get('a_contains', [])):
        faithful_hits += 1

print({
    "N": n,
    f"Recall@{args.k}(doc-hint)": recall_hits/n,
    "Faithfulness(proxy)": faithful_hits/n,
})
