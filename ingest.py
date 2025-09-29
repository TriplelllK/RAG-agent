import os, argparse, json
from typing import List, Dict
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re

DOC_MAP = {
    "avarii_i_signalizacii_u300.pdf": "Аварии и сигнализации У-300 КТЛ-1",
    "normy_tekh_rezhima_u300.pdf": "Нормы технологического режима У-300 КТЛ-1",
    "tekhnologicheskiy_reglament_u300.pdf": "Технологический регламент У-300 КТЛ-1",
}

def chunk_text(text: str, chunk: int = 900, overlap: int = 150) -> List[str]:
    """
    Новый способ: сначала режем по абзацам, потом fallback по chunk.
    Это позволяет сохранять целые секции вроде 'Насосы G-304'.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) <= chunk:
            out.append(p)
        else:
            i = 0
            while i < len(p):
                out.append(p[i:i+chunk])
                i += (chunk - overlap)
    return out

def load_pdf(path: str) -> List[Dict]:
    reader = PdfReader(path)
    chunks = []
    base = os.path.basename(path)
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        for part in chunk_text(txt):
            chunks.append({
                "doc_name": DOC_MAP.get(base, base),
                "page": i,
                "text": part.strip()
            })
    return chunks

def main(data_dir: str, out_dir: str, chunk: int, overlap: int):
    os.makedirs(out_dir, exist_ok=True)
    all_chunks = []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(data_dir, fname)
        all_chunks.extend(load_pdf(path))
    # эмбеддинги
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64)
    embs = np.asarray(embs, dtype='float32')
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, os.path.join(out_dir, 'faiss.index'))
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print(f"Indexed {len(all_chunks)} chunks → {out_dir}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--chunk', type=int, default=900)
    ap.add_argument('--overlap', type=int, default=150)
    args = ap.parse_args()
    main(args.data_dir, args.out_dir, args.chunk, args.overlap)
