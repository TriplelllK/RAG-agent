import os, json, regex as re, pdfplumber

PAT = re.compile(r"\b([A-Z]{2,4}-\d{3,}\b|LIC-\d+|PDT-\d+|PDI-\d+|FT-\d+)", re.I)

DOC_MAP = {
    "avarii_i_signalizacii_u300.pdf": "Аварии и сигнализации У-300 КТЛ-1",
    "normy_tekh_rezhima_u300.pdf": "Нормы технологического режима У-300 КТЛ-1",
    "tekhnologicheskiy_reglament_u300.pdf": "Технологический регламент У-300 КТЛ-1",
}

def extract_tables_from_pdf(path: str):
    out = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for tb in tables or []:
                rows = [[(cell or "").strip() for cell in row] for row in tb]
                for r in rows:
                    text = " | ".join(r)
                    for m in PAT.findall(text):
                        out.append({
                            "tag": m.upper(),
                            "row": r,
                            "page": i,
                            "doc_basename": os.path.basename(path)
                        })
    return out

def build_kv(data_dir: str, out_path: str):
    kv = {}
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        full = os.path.join(data_dir, fname)
        for it in extract_tables_from_pdf(full):
            entry = {
                "doc_name": DOC_MAP.get(fname, fname),
                "page": it["page"],
                "row": it["row"]
            }
            kv.setdefault(it["tag"], []).append(entry)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kv, f, ensure_ascii=False)
    print(f"Saved {len(kv)} tags → {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="storage/tables_kv.json")
    args = ap.parse_args()
    build_kv(args.data_dir, args.out)
