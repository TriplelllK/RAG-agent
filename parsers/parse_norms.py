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

def parse_norm_row(cells: List[str], equip_hint: str, page_num: int) -> Optional[Norm]:
    if not cells or len(cells) < 4:
        return None

    instr = cells[0].strip()
    param = cells[1].strip() if len(cells) > 1 else ""
    unit  = cells[2].strip() if len(cells) > 2 else ""

    if not instr or not param:
        return None

    # допустимые диапазоны
    range_min = None
    range_max = None
    work_min  = None
    work_max  = None

    def _to_float(x: str) -> Optional[float]:
        try:
            return float(x.replace(",", "."))
        except:
            return None

    if len(cells) > 3 and cells[3] not in ["", "-", "—"]:
        range_min = _to_float(cells[3])
    if len(cells) > 4 and cells[4] not in ["", "-", "—"]:
        range_max = _to_float(cells[4])
    if len(cells) > 5 and cells[5] not in ["", "-", "—"]:
        work_min  = _to_float(cells[5])
    if len(cells) > 6 and cells[6] not in ["", "-", "—"]:
        work_max  = _to_float(cells[6])

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
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                table = page.extract_table()
            except:
                table = None
            if not table:
                continue

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
    return norms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = [n.__dict__ for n in parse_norms(args.pdf)]
    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Parsed {len(data)} rows → {args.out}")
