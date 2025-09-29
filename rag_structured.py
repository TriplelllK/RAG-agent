import json, os
from typing import List
from dataclasses import dataclass

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

@dataclass
class Alarm:
    equipment: str
    instrument: str
    param: str
    unit: str
    setpoint: str|None
    action: str
    note: str
    page: int

def load_norms(path="storage/norms.json") -> List[Norm]:
    if not os.path.exists(path): return []
    data = json.load(open(path, "r", encoding="utf-8"))
    return [Norm(**r) for r in data]

def load_alarms(path="storage/alarms.json") -> List[Alarm]:
    if not os.path.exists(path): return []
    data = json.load(open(path, "r", encoding="utf-8"))
    return [Alarm(**r) for r in data]

# --- поиск по оборудованию ---
def norms_by_equipment(norms: List[Norm], equipment: str) -> List[Norm]:
    eq = (equipment or "").upper()
    return [n for n in norms if n.equipment.upper()==eq]

def alarms_by_equipment(alarms: List[Alarm], equipment: str) -> List[Alarm]:
    eq = (equipment or "").upper()
    return [a for a in alarms if a.equipment.upper()==eq]

# --- поиск по приборам ---
def find_norm_by_instrument(norms: List[Norm], instrument: str) -> List[Norm]:
    inst = (instrument or "").upper().strip()
    return [n for n in norms if n.instrument.upper() == inst]

def find_alarm_by_instrument(alarms: List[Alarm], instrument: str) -> List[Alarm]:
    inst = (instrument or "").upper().strip()
    return [a for a in alarms if a.instrument.upper() == inst]

# --- форматирование ---
def format_norm_line(n: Norm) -> str:
    parts = []
    head = f"{n.instrument}: {n.param}".strip(": ")
    if n.unit: head += f" ({n.unit})"
    parts.append(head)
    rng = []
    if n.range_min is not None or n.range_max is not None:
        rng.append(f"допустимый диапазон: {n.range_min}–{n.range_max}")
    if n.work_min is not None or n.work_max is not None:
        rng.append(f"рабочий диапазон: {n.work_min}–{n.work_max}")
    if rng:
        parts.append("  ▸ " + "; ".join(rng))
    parts.append(f"  ▸ стр. {n.page}")
    return "\n".join(parts)

def format_alarm_line(a: Alarm) -> str:
    if not a.action or a.action.strip() in ["-", "—"]:
        return ""
    head = f"{a.instrument}: {a.param}".strip(": ")
    if a.unit:
        head += f" ({a.unit})"
    if a.setpoint:
        head += f", уставка {a.setpoint}"
    line = f"Сигнализация по {head}. "
    line += f"При срабатывании выполняется действие: {a.action}. "
    if a.note and a.note.strip() not in ["-", "—"]:
        line += f"Примечание: {a.note}. "
    line += f"(стр. {a.page})"
    return line
