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

def parse_alarm_row(cells: List[str], equip_hint: str, page_num: int) -> Optional[Alarm]:
    """
    Разбор одной строки таблицы аварий.
    Берём прибор, параметр, единицы, уставку и действие.
    """
    if not cells or len(cells) < 3:
        return None

    instr = cells[0].strip()
    param = cells[1].strip()
    unit  = cells[2].strip()

    if not instr or not param:
        return None

    # ищем уставку в первых колонках после единиц
    setpoint = None
    for c in cells[3:6]:
        if c and c not in ["-", "—"]:
            setpoint = c.strip()
            break

    # ищем действие в последних ячейках
    action = ""
    for c in reversed(cells):
        if any(word in c for word in ["Закрытие", "Открытие", "Пуск", "Остановка", "Перевод"]):
            action = c.strip()
            break

    # примечание (если оно есть и не "-")
    note = ""
    if cells and cells[-1] not in ["", "-", "—"]:
        note = cells[-1].strip()

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

                # обновляем шапку оборудования
                eq = _maybe_equipment(" ".join(cells))
                if eq:
                    equip_hint = eq
                    continue

                alarm = parse_alarm_row(cells, equip_hint, page_num)
                if alarm:
                    alarms.append(alarm)
    return alarms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = [a.__dict__ for a in parse_alarms(args.pdf)]
    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Parsed {len(data)} rows → {args.out}")
