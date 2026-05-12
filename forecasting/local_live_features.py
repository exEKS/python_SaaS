from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from forecasting.default_feature_row import DEMO_BATCH_DICT
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


def stats_dir() -> Path:
    d = local_live_root() / "stats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def hourly_stats_path() -> Path:
    return stats_dir() / "hourly_alert_stats.json"


def weather_history_path() -> Path:
    """Visual Crossing snapshot written by `forecasting/weather_collector.py`."""
    return local_live_root() / "weather_history.json"


# City keys in `weather_history.json` match Visual Crossing timeline names
# (see `forecasting/weather_collector.py` REGIONS).
_WEATHER_VC_CITIES: frozenset[str] = frozenset(
    {
        "Kyiv",
        "Vinnytsia",
        "Lutsk",
        "Dnipro",
        "Donetsk",
        "Zhytomyr",
        "Uzhhorod",
        "Zaporizhzhia",
        "Ivano-Frankivsk",
        "Kropyvnytskyi",
        "Luhansk",
        "Lviv",
        "Mykolaiv",
        "Odesa",
        "Poltava",
        "Rivne",
        "Sumy",
        "Ternopil",
        "Kharkiv",
        "Kherson",
        "Khmelnytskyi",
        "Cherkasy",
        "Chernivtsi",
        "Chernihiv",
        "Simferopol",
    }
)

# Map API / UI region tokens → Visual Crossing city key in `weather_history.json`
_API_TOKEN_TO_VC_CITY: dict[str, str] = {
    "kyiv": "Kyiv",
    "kiev": "Kyiv",
    "kharkiv": "Kharkiv",
    "lviv": "Lviv",
    "odesa": "Odesa",
    "odessa": "Odesa",
    "dnipro": "Dnipro",
    "dnipropetrovsk": "Dnipro",
    "zaporizhzhia": "Zaporizhzhia",
    "donetsk": "Donetsk",
    "vinnytsia": "Vinnytsia",
    "chernihiv": "Chernihiv",
    "cherkasy": "Cherkasy",
    "chernivtsi": "Chernivtsi",
    "ivano-frankivsk": "Ivano-Frankivsk",
    "kherson": "Kherson",
    "khmelnytskyi": "Khmelnytskyi",
    "kirovohrad": "Kropyvnytskyi",
    "kropyvnytskyi": "Kropyvnytskyi",
    "luhansk": "Luhansk",
    "mykolaiv": "Mykolaiv",
    "poltava": "Poltava",
    "rivne": "Rivne",
    "sumy": "Sumy",
    "ternopil": "Ternopil",
    "zhytomyr": "Zhytomyr",
    "zakarpattia": "Uzhhorod",
    "uzhhorod": "Uzhhorod",
    "volyn": "Lutsk",
    "lutsk": "Lutsk",
    "crimea": "Simferopol",
    "sevastopol": "Simferopol",
}

_WEATHER_COLS: frozenset[str] = frozenset(
    k for k in DEMO_BATCH_DICT if k.startswith("weather_")
)

_WH_JSON_CACHE: tuple[float | None, dict | None] = (None, None)


def _read_weather_history_json() -> dict | None:
    path = weather_history_path()
    if not path.is_file():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    global _WH_JSON_CACHE
    if _WH_JSON_CACHE[0] == mtime and isinstance(_WH_JSON_CACHE[1], dict):
        return _WH_JSON_CACHE[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    _WH_JSON_CACHE = (mtime, data)
    return data


def visual_crossing_city_for_region(region: str) -> str | None:
    """Resolve API `region` string to a key present in `weather_history.json`."""
    raw = (region or "").strip()
    if not raw:
        return None
    if raw in _WEATHER_VC_CITIES:
        return raw
    low = raw.lower()
    if low in _API_TOKEN_TO_VC_CITY:
        return _API_TOKEN_TO_VC_CITY[low]
    slug = re.sub(r"[^a-z0-9]+", "-", low).strip("-")
    if slug in _API_TOKEN_TO_VC_CITY:
        return _API_TOKEN_TO_VC_CITY[slug]
    for token, city in _API_TOKEN_TO_VC_CITY.items():
        if token and (token in slug or token in low):
            return city
    for city in _WEATHER_VC_CITIES:
        if city.lower() == low:
            return city
    return None


def _weather_history_pick_date(history: dict, date_part: str) -> str | None:
    dates = sorted(
        k
        for k in history
        if isinstance(k, str) and len(k) >= 10 and k[0:4].isdigit() and k[4] == "-"
    )
    if not dates:
        return None
    if date_part in history and isinstance(history.get(date_part), dict):
        return date_part
    past = [d for d in dates if d <= date_part]
    if past:
        return past[-1]
    return dates[-1]


def _float_or_none(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _conditions_from_region_blob(blob: dict) -> str:
    c = blob.get("conditions")
    if isinstance(c, str) and c.strip():
        return c.strip()
    hours = blob.get("hours")
    if isinstance(hours, list):
        for h in hours:
            if isinstance(h, dict):
                hc = h.get("conditions")
                if isinstance(hc, str) and hc.strip():
                    return hc.strip()
    return ""


def _weather_one_hot_overrides(conditions: str) -> dict[str, float]:
    """Match Visual Crossing `conditions` text to model one-hot `weather_*` columns."""
    out: dict[str, float] = {c: 0.0 for c in _WEATHER_COLS}
    if not conditions:
        out["weather_Overcast"] = 1.0
        return out

    segments = [s.strip() for s in conditions.split(",") if s.strip()]
    chosen: str | None = None
    for seg in segments:
        token = re.sub(r"\s+", "_", seg.strip())
        col = f"weather_{token}"
        if col in out:
            chosen = col
            break
    if chosen is None:
        low = conditions.lower()
        for col in sorted(out.keys(), key=len, reverse=True):
            suf = col.replace("weather_", "").replace("_", " ").lower()
            if len(suf) >= 4 and suf in low:
                chosen = col
                break
    if chosen is None:
        out["weather_Overcast"] = 1.0
    else:
        out[chosen] = 1.0
    return out


def weather_feature_overrides_for_prediction(
    region: str, date_part: str
) -> tuple[dict[str, float], dict]:
    """
    Build feature overrides from `weather_history.json` (Visual Crossing snapshot).

    Fills day_temp / day_tempmax / day_tempmin / day_humidity / day_windspeed and
    `weather_*` one-hots when the file and region city exist.
    """
    if os.environ.get("WARWATCH_DISABLE_WEATHER_HISTORY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return {}, {}

    city = visual_crossing_city_for_region(region)
    if city is None:
        return {}, {}

    history = _read_weather_history_json()
    if not history:
        return {}, {}

    picked = _weather_history_pick_date(history, date_part)
    if not picked:
        return {}, {}

    day_entry = history.get(picked)
    if not isinstance(day_entry, dict):
        return {}, {}

    regions = day_entry.get("regions")
    if not isinstance(regions, dict):
        return {}, {}

    blob = regions.get(city)
    if not isinstance(blob, dict):
        return {}, {}

    overrides: dict[str, float] = {}

    t = _float_or_none(blob.get("temp"))
    if t is not None:
        overrides["day_temp"] = t

    h = _float_or_none(blob.get("humidity"))
    if h is not None:
        overrides["day_humidity"] = h

    ws = _float_or_none(blob.get("windspeed"))
    if ws is not None:
        overrides["day_windspeed"] = ws

    tmax = _float_or_none(blob.get("tempmax")) or _float_or_none(blob.get("tempMax"))
    tmin = _float_or_none(blob.get("tempmin")) or _float_or_none(blob.get("tempMin"))
    hours = blob.get("hours")
    hour_temps: list[float] = []
    if isinstance(hours, list):
        for hr in hours:
            if isinstance(hr, dict):
                tv = _float_or_none(hr.get("temp"))
                if tv is not None:
                    hour_temps.append(tv)
    if tmax is None and hour_temps:
        tmax = max(hour_temps)
    if tmin is None and hour_temps:
        tmin = min(hour_temps)
    if tmax is not None:
        overrides["day_tempmax"] = tmax
    if tmin is not None:
        overrides["day_tempmin"] = tmin

    cond = _conditions_from_region_blob(blob)
    overrides.update(_weather_one_hot_overrides(cond))

    meta = {
        "weather_history_file": weather_history_path().name,
        "weather_date": picked,
        "weather_city": city,
    }
    if cond:
        meta["weather_conditions"] = cond[:120]
    return overrides, meta


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


def _canonical_regions() -> list[str]:
    return sorted(REGION_ALARM_KEYWORDS.keys())


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


def _iter_alert_files_for_date(date_str: str) -> list[Path]:
    d = alerts_dir()
    if not d.is_dir():
        return []
    prefix = f"alerts_{date_str}_"
    return sorted(
        (p for p in d.glob(f"{prefix}*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )


def _extract_hour_from_alert_filename(path: Path) -> int | None:
    m = re.match(r"alerts_(\d{4}-\d{2}-\d{2})_(\d{2})-\d{2}-\d{2}\.json$", path.name)
    if not m:
        return None
    try:
        h = int(m.group(2))
    except ValueError:
        return None
    if 0 <= h <= 23:
        return h
    return None


def update_hourly_stats_from_payload(data: object, ts: datetime) -> None:
    """
    Persist compact hourly priors from each alerts snapshot.
    This survives raw-json purge and allows hourly profile shaping.
    """
    if not isinstance(data, list):
        return
    p = hourly_stats_path()
    if p.is_file():
        try:
            stats = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats = {}
    else:
        stats = {}
    if not isinstance(stats, dict):
        stats = {}
    stats.setdefault("version", 1)
    stats.setdefault("hours", {})
    stats.setdefault("regions", _canonical_regions())
    hours = stats["hours"]

    hk = str(int(ts.hour))
    bucket = hours.setdefault(
        hk,
        {
            "samples": 0,
            "national_alert_ratio_sum": 0.0,
            "regions": {k: {"active": 0, "samples": 0} for k in _canonical_regions()},
        },
    )
    bucket["samples"] = int(bucket.get("samples", 0)) + 1

    total_rows = 0
    active_rows = 0
    region_active_flags = {k: False for k in _canonical_regions()}

    for row in data:
        if not isinstance(row, dict):
            continue
        total_rows += 1
        name = str(row.get("regionName") or row.get("name") or "")
        alerts = row.get("activeAlerts")
        if alerts is None:
            alerts = row.get("alerts")
        is_active = bool(alerts)
        if is_active:
            active_rows += 1
        for key in region_active_flags:
            if _api_name_matches_region(name, key):
                region_active_flags[key] = region_active_flags[key] or is_active

    if total_rows > 0:
        bucket["national_alert_ratio_sum"] = float(
            bucket.get("national_alert_ratio_sum", 0.0) + (active_rows / float(total_rows))
        )
    for key, is_active in region_active_flags.items():
        rs = bucket["regions"].setdefault(key, {"active": 0, "samples": 0})
        rs["samples"] = int(rs.get("samples", 0)) + 1
        if is_active:
            rs["active"] = int(rs.get("active", 0)) + 1

    stats["updated_at"] = ts.isoformat(timespec="seconds")
    p.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def hourly_alert_profile_context(region: str, date_iso: str) -> dict:
    """
    Build hourly priors for a region:
    - historical priors from persisted hourly stats
    - today's observed active flags from raw snapshots
    """
    region_key = _region_match_key(region)
    region_prior = [None] * 24
    national_prior = [None] * 24

    sp = hourly_stats_path()
    if sp.is_file():
        try:
            stats = json.loads(sp.read_text(encoding="utf-8"))
            hours = stats.get("hours", {})
            for h in range(24):
                b = hours.get(str(h))
                if not isinstance(b, dict):
                    continue
                samples = int(b.get("samples", 0))
                if samples > 0:
                    national_prior[h] = float(b.get("national_alert_ratio_sum", 0.0)) / float(samples)
                rb = (b.get("regions") or {}).get(region_key)
                if isinstance(rb, dict):
                    rs = int(rb.get("samples", 0))
                    if rs > 0:
                        region_prior[h] = float(rb.get("active", 0)) / float(rs)
        except (OSError, json.JSONDecodeError):
            pass

    date_part = (date_iso or "")[:10]
    today_observed = [None] * 24
    for f in _iter_alert_files_for_date(date_part):
        h = _extract_hour_from_alert_filename(f)
        if h is None:
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        is_active = False
        for row in payload:
            if not isinstance(row, dict):
                continue
            name = str(row.get("regionName") or row.get("name") or "")
            alerts = row.get("activeAlerts")
            if alerts is None:
                alerts = row.get("alerts")
            if _api_name_matches_region(name, region_key):
                is_active = bool(alerts)
                break
        today_observed[h] = 1.0 if is_active else 0.0

    return {
        "region_key": region_key,
        "region_prior": region_prior,
        "national_prior": national_prior,
        "today_observed": today_observed,
    }


def live_feature_overrides_for_prediction(
    region: str,
    date_iso: str,
) -> tuple[dict[str, float], dict]:
    """
    Build feature overrides from ISW + Ukraine Alarm JSON snapshots on disk,
    plus `weather_history.json` (Visual Crossing) when available.
    Explicit API query overrides should be merged after this (they win).
    """
    date_part = (date_iso or "")[:10]
    if len(date_part) != 10:
        return {}, {}

    meta: dict = {}
    overrides: dict[str, float] = {}

    if os.environ.get("WARWATCH_DISABLE_LOCAL_LIVE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
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

    w_over, w_meta = weather_feature_overrides_for_prediction(region, date_part)
    if w_over:
        overrides.update(w_over)
    if w_meta:
        meta["weather"] = w_meta

    return overrides, meta
