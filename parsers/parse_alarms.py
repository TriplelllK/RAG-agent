import argparse, json, pdfplumber, re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Alarm:
    equipment: str
    instrument: str
    param: str
    unit: str
    setpoint: Optional[str]
    action: str
    note: str
    page: int

def _maybe_equipment(text: str) -> Optional[str]:
    text = text.strip()
    if re.match(r"^[FGTD]-\d{3,4}", text):
        return text.split()[0].strip(",.;:")
    return None

def _is_instrument(value: str) -> bool:
    if not value:
        return False
    return bool(re.match(r"^(?:[A-Z]{2,5}-?\d{3,6}(?:_\d+)?)$", value.strip().upper()))

def _is_setpoint_like(value: str) -> bool:
    if not value:
        return False
    v = value.strip()
    if v in ["-", "—"]:
        return False
    if re.search(r"[A-Za-zА-Яа-я]", v):
        return False
    return bool(re.match(r"^[\d\s,\.\-+/<>=%]+$", v))

def parse_alarm_row(cells: List[str], equip_hint: str, page_num: int) -> Optional[Alarm]:
    # Разбираем одну строку таблицы аварий.
    cells = [c.strip() for c in cells if c and c.strip()]
    if not cells:
        return None

    instr_idx = -1
    for idx, cell in enumerate(cells):
        if _is_instrument(cell):
            instr_idx = idx
            break
    if instr_idx == -1:
        return None

    instr = cells[instr_idx].strip()
    tail = cells[instr_idx + 1:]
    param = tail[0].strip() if len(tail) > 0 else ""
    unit = tail[1].strip() if len(tail) > 1 else ""

    if not instr or not param:
        return None

    # Ищем уставку рядом с единицами.
    setpoint = None
    for c in tail[:6]:
        if _is_setpoint_like(c):
            setpoint = c.strip()
            break

    # Ищем действие ближе к концу строки.
    action = ""
    for c in reversed(tail):
        if any(word in c for word in ["Закрытие", "Открытие", "Пуск", "Остановка", "Перевод"]):
            action = c.strip()
            break

    # Берем примечание, если оно есть.
    note = ""
    if tail and tail[-1] not in ["", "-", "—"]:
        note = tail[-1].strip()

    return Alarm(
        equipment=equip_hint or "",
        instrument=instr,
        param=param,
        unit=unit,
        setpoint=setpoint,
        action=action,
        note=note,
        page=page_num
    )

def parse_alarms(pdf_path: str) -> List[Alarm]:
    alarms = []
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

                    # Обновляем текущее оборудование.
                    eq = _maybe_equipment(" ".join(cells))
                    if eq:
                        equip_hint = eq
                        continue

                    alarm = parse_alarm_row(cells, equip_hint, page_num)
                    if alarm:
                        alarms.append(alarm)
                    else:
                        skipped_rows += 1

    print(f"Skipped rows: {skipped_rows}")
    return alarms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = [a.__dict__ for a in parse_alarms(args.pdf)]
    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Parsed rows: {len(data)}")
