import os, argparse, json
from typing import List, Dict
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re

STATE_FILE = "ingest_state.json"
SUPPORTED_EXTS = {".pdf", ".txt", ".md"}

DOC_MAP = {
    "avarii_i_signalizacii_u300.pdf": "Аварии и сигнализации У-300 КТЛ-1",
    "normy_tekh_rezhima_u300.pdf": "Нормы технологического режима У-300 КТЛ-1",
    "tekhnologicheskiy_reglament_u300.pdf": "Технологический регламент У-300 КТЛ-1",
    "reglament_sample.txt": "Технологический регламент У-300 КТЛ-1 (sample)",
    "norms_sample.txt": "Нормы технологического режима У-300 КТЛ-1 (sample)",
    "alarms_sample.txt": "Аварии и сигнализации У-300 КТЛ-1 (sample)",
}

def chunk_text(text: str, chunk: int = 900, overlap: int = 150) -> List[str]:
    # Сначала делим по абзацам, потом режем длинные куски.
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
                i += max(1, (chunk - overlap))
    return out

def load_pdf(path: str, chunk: int = 900, overlap: int = 150) -> List[Dict]:
    reader = PdfReader(path)
    chunks = []
    base = os.path.basename(path)
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        for part in chunk_text(txt, chunk=chunk, overlap=overlap):
            chunks.append({
                "doc_name": DOC_MAP.get(base, base),
                "page": i,
                "text": part.strip()
            })
    return chunks

def load_text_file(path: str, chunk: int = 900, overlap: int = 150) -> List[Dict]:
    base = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    chunks = []
    for part in chunk_text(txt, chunk=chunk, overlap=overlap):
        chunks.append({
            "doc_name": DOC_MAP.get(base, base),
            "page": 1,
            "text": part.strip()
        })
    return chunks

def _file_signatures(data_dir: str) -> Dict[str, Dict]:
    signatures = {}
    for fname in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        full = os.path.join(data_dir, fname)
        st = os.stat(full)
        signatures[fname] = {
            "size": int(st.st_size),
            "mtime": int(st.st_mtime),
        }
    return signatures

def _load_state(out_dir: str) -> Dict:
    path = os.path.join(out_dir, STATE_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_state(out_dir: str, state: Dict) -> None:
    with open(os.path.join(out_dir, STATE_FILE), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def main(data_dir: str, out_dir: str, chunk: int, overlap: int, force: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    signatures = _file_signatures(data_dir)
    prev = _load_state(out_dir)
    current_cfg = {
        "chunk": chunk,
        "overlap": overlap,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "files": signatures,
    }

    idx_path = os.path.join(out_dir, 'faiss.index')
    meta_path = os.path.join(out_dir, 'meta.json')
    if (not force) and os.path.exists(idx_path) and os.path.exists(meta_path) and prev == current_cfg:
        print("Index is up to date")
        return

    all_chunks = []
    for fname in os.listdir(data_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        path = os.path.join(data_dir, fname)
        if ext == ".pdf":
            all_chunks.extend(load_pdf(path, chunk=chunk, overlap=overlap))
        else:
            all_chunks.extend(load_text_file(path, chunk=chunk, overlap=overlap))

    if not all_chunks:
        print("No chunks found")
        return

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64)
    embs = np.asarray(embs, dtype='float32')
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, idx_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    _save_state(out_dir, current_cfg)
    print(f"Indexed chunks: {len(all_chunks)}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--chunk', type=int, default=900)
    ap.add_argument('--overlap', type=int, default=150)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    main(args.data_dir, args.out_dir, args.chunk, args.overlap, force=args.force)
