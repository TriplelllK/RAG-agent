import argparse, json, pdfplumber, re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Norm:
    instrument: str
    param: str
    unit: str
    range_min: float|None
    range_max: float|None
    work_min: float|None
    work_max: float|None
    page: int
    equipment: str

def _maybe_equipment(text: str) -> Optional[str]:
    text = text.strip()
    if re.match(r"^[FGTD]-\d{3,4}", text):
        return text.replace(" ", "").replace("–", "-")
    return None

def _is_instrument(value: str) -> bool:
    if not value:
        return False
    return bool(re.match(r"^(?:[A-Z]{2,5}-?\d{3,6}(?:_\d+)?)$", value.strip().upper()))

def _is_number_like(value: str) -> bool:
    if not value:
        return False
    value = value.strip().replace(",", ".")
    return bool(re.match(r"^-?\d+(?:\.\d+)?$", value))

def _to_float(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", "."))
    except Exception:
        return None

def parse_norm_row(cells: List[str], equip_hint: str, page_num: int) -> Optional[Norm]:
    cells = [c.strip() for c in cells if c and c.strip()]
    if not cells:
        return None

    instr_idx = -1
    for idx, cell in enumerate(cells):
        if _is_instrument(cell):
            instr_idx = idx
            break
    if instr_idx == -1:
        instr_idx = 0

    instr = cells[instr_idx].strip()
    if not _is_instrument(instr):
        return None

    tail = cells[instr_idx + 1:]
    text_tail = [x for x in tail if not _is_number_like(x)]
    param = text_tail[0].strip() if text_tail else ""
    unit = text_tail[1].strip() if len(text_tail) > 1 else ""

    if not instr or not param:
        return None

    # Читаем диапазоны.
    range_min = None
    range_max = None
    work_min  = None
    work_max  = None

    num_tail = [x for x in tail if _is_number_like(x)]
    if len(num_tail) >= 1:
        range_min = _to_float(num_tail[0])
    if len(num_tail) >= 2:
        range_max = _to_float(num_tail[1])
    if len(num_tail) >= 3:
        work_min = _to_float(num_tail[2])
    if len(num_tail) >= 4:
        work_max = _to_float(num_tail[3])

    return Norm(
        equipment=equip_hint or "",
        instrument=instr,
        param=param,
        unit=unit,
        range_min=range_min,
        range_max=range_max,
        work_min=work_min,
        work_max=work_max,
        page=page_num
    )

def parse_norms(pdf_path: str) -> List[Norm]:
    norms = []
    equip_hint = None
    skipped_rows = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            if not tables:
                continue

            for table in tables:
                for row in table:
                    if not row:
                        continue
                    cells = [c.strip() if c else "" for c in row]

                    eq = _maybe_equipment(" ".join(cells))
                    if eq:
                        equip_hint = eq
                        continue

                    norm = parse_norm_row(cells, equip_hint, page_num)
                    if norm:
                        norms.append(norm)
                    else:
                        skipped_rows += 1

    print(f"Skipped rows: {skipped_rows}")
    return norms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = [n.__dict__ for n in parse_norms(args.pdf)]
    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Parsed rows: {len(data)}")
