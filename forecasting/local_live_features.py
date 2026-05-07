from __future__ import annotations

import json
import os
import re
from pathlib import Path

from forecasting.paths import ROOT

HIGH_SIGNAL_TERMS = (
    "missile",
    "drone",
    "strike",
    "shelling",
    "offensive",
    "rao",
    "ракет",
    "дрон",
    "удар",
    "обстріл",
    "наступ",
    "вибух",
    "авіаці",
    "балістич",
    "s-300",
    "shahed",
)


def local_live_root() -> Path:
    override = os.environ.get("WARWATCH_LOCAL_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (ROOT / "data" / "local_live").resolve()


def isw_dir() -> Path:
    d = local_live_root() / "isw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def alerts_dir() -> Path:
    d = local_live_root() / "alerts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def purge_local_live_storage() -> dict[str, int]:
    """Remove collected JSON files under local_live (keeps directory layout)."""
    root = local_live_root()
    removed = 0
    for sub in ("isw", "alerts"):
        p = root / sub
        if not p.is_dir():
            continue
        for f in p.glob("*.json"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return {"removed_files": removed, "root": str(root)}


# Keywords to match API regionName (lowercase) to prediction region names (English)
REGION_ALARM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "kyiv": ("київ", "kyiv", "kiev", "м.київ"),
    "kharkiv": ("харків", "kharkiv"),
    "lviv": ("львів", "lviv"),
    "odesa": ("одес", "odesa", "odessa"),
    "dnipropetrovsk": ("дніпр", "dnipro", "dnipropetrovsk"),
    "zaporizhzhia": ("запоріж", "zaporizhzhia", "zaporizh"),
    "donetsk": ("донец", "donetsk"),
    "vinnytsia": ("вінниц", "vinnyts"),
    "chernihiv": ("чернігів", "chernihiv", "chernigov"),
    "cherkasy": ("черкас", "cherkasy"),
    "chernivtsi": ("чернів", "chernivtsi"),
    "ivano-frankivsk": ("івано-франк", "ivano-frankivsk", "ivano frankivsk"),
    "kherson": ("херсон", "kherson"),
    "khmelnytskyi": ("хмельниц", "khmelnyts"),
    "kirovohrad": ("кіровоград", "kirovohrad", "кропив", "kropyvnytskyi"),
    "luhansk": ("луганськ", "luhansk", "луган"),
    "mykolaiv": ("миколаїв", "mykolaiv", "nikolaev"),
    "poltava": ("полтав", "poltava"),
    "rivne": ("рівн", "rivne"),
    "sumy": ("сум", "sumy"),
    "ternopil": ("терноп", "ternopil"),
    "zhytomyr": ("житомир", "zhytomyr"),
    "zakarpattia": ("закарпат", "zakarpatt", "uzhhorod"),
    "volyn": ("волин", "volyn", "lutsk"),
    "crimea": ("крим", "crimea", "сімферопол", "simferopol"),
    "sevastopol": ("севастопол", "sevastopol"),
}


def _region_match_key(region: str) -> str:
    s = region.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _api_name_matches_region(api_region_name: str, predict_region: str) -> bool:
    nm = (api_region_name or "").lower()
    key = _region_match_key(predict_region)
    if key in REGION_ALARM_KEYWORDS:
        return any(k in nm for k in REGION_ALARM_KEYWORDS[key])
    if key.replace("-", "") in nm.replace(" ", "").replace("-", ""):
        return True
    return key in nm.replace(" ", "-")


def _isw_text_intensity(text: str) -> float:
    if not text or not text.strip():
        return 0.12
    low = text.lower()
    n = len(text)
    length_part = min(1.0, n / 90000.0)
    hits = sum(1 for t in HIGH_SIGNAL_TERMS if t in low)
    score = 0.12 + 0.45 * length_part + min(0.4, hits * 0.035)
    return float(max(0.0, min(1.0, score)))


def _duration_from_alert_item(item: object) -> float:
    if not isinstance(item, dict):
        return 45.0
    for key in ("duration", "durationMinutes", "duration_minutes", "lengthMinutes"):
        v = item.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 45.0


def _summarize_alerts_payload(data: object, predict_region: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(data, list):
        return out

    regions_with_alert = 0
    total_duration = 0.0
    this_region_active = False

    for row in data:
        if not isinstance(row, dict):
            continue
        name = str(row.get("regionName") or row.get("name") or "")
        alerts = row.get("activeAlerts")
        if alerts is None:
            alerts = row.get("alerts")
        if not alerts:
            continue
        if not isinstance(alerts, (list, tuple)):
            alerts = [alerts]
        regions_with_alert += 1
        for a in alerts:
            total_duration += _duration_from_alert_item(a)
        if _api_name_matches_region(name, predict_region):
            this_region_active = True

    n = len(data) if isinstance(data, list) else 1
    out["alarm_count"] = float(regions_with_alert)
    out["alarm_all_region"] = 1.0 if regions_with_alert > 0 else 0.0
    out["alarm_total_duration_min"] = float(total_duration)
    out["alarm_lag1"] = 1.0 if this_region_active else 0.0
    tension = regions_with_alert / max(1.0, float(n))
    out["alarm_roll7_mean"] = float(max(0.0, min(1.0, 0.15 + 0.65 * tension)))
    return out


def _latest_alerts_file_for_date(date_str: str) -> Path | None:
    d = alerts_dir()
    if not d.is_dir():
        return None
    prefix = f"alerts_{date_str}_"
    candidates = sorted(
        (p for p in d.glob(f"{prefix}*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def live_feature_overrides_for_prediction(
    region: str,
    date_iso: str,
) -> tuple[dict[str, float], dict]:
    """
    Build feature overrides from ISW + Ukraine Alarm JSON snapshots on disk.
    Explicit API query overrides should be merged after this (they win).
    """
    if os.environ.get("WARWATCH_DISABLE_LOCAL_LIVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return {}, {}

    date_part = (date_iso or "")[:10]
    if len(date_part) != 10:
        return {}, {}

    meta: dict = {}
    overrides: dict[str, float] = {}

    isw_path = isw_dir() / f"isw_data_{date_part}.json"
    if isw_path.is_file():
        try:
            payload = json.loads(isw_path.read_text(encoding="utf-8"))
            text = str(payload.get("text") or "")
            overrides["text_intensity_index"] = _isw_text_intensity(text)
            meta["isw"] = isw_path.name
        except (OSError, json.JSONDecodeError):
            pass

    alerts_path = _latest_alerts_file_for_date(date_part)
    if alerts_path is not None:
        try:
            payload = json.loads(alerts_path.read_text(encoding="utf-8"))
            alarm_feats = _summarize_alerts_payload(payload, region)
            overrides.update(alarm_feats)
            meta["alerts"] = alerts_path.name
        except (OSError, json.JSONDecodeError):
            pass

    return overrides, meta
