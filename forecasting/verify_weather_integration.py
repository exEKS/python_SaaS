"""
Перевірка інтеграції weather_history.json з прогнозом.

За замовчуванням: реальна модель з model_dir() і справжній файл
`data/local_live/weather_history.json` (як на сервері / у репо).

Режим --probe: тимчасова модель + фейковий JSON (як раніше, без .pkl).

З кореня репозиторію:
  python forecasting/verify_weather_integration.py
  python forecasting/verify_weather_integration.py --region Kharkiv --date 2026-05-12

Каталог із моделлю (варіант B): змінна WARWATCH_MODEL_DIR або рядок у .env
(див. повідомлення скрипта, якщо .pkl не знайдено).

  python forecasting/verify_weather_integration.py --probe
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_WARWATCH_ENV_KEYS = (
    "WARWATCH_MODEL_DIR",
    "WARWATCH_MODEL_ALARM",
    "WARWATCH_LOCAL_DATA_DIR",
)


def _bootstrap_env_from_repo_dotenv() -> None:
    """Підхопити WARWATCH_* з .env (dotenv або простий парсер, без перезапису вже заданих у shell)."""
    env_path = _REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass

    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if key not in _WARWATCH_ENV_KEYS:
            continue
        if os.environ.get(key, "").strip():
            continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        os.environ[key] = val


def _model_dir_diagnostics() -> str:
    """Пояснення, якщо WARWATCH_MODEL_DIR заданий, але .pkl не видно."""
    raw = os.environ.get("WARWATCH_MODEL_DIR", "").strip()
    if not raw:
        return ""
    p = Path(raw).expanduser()
    if not p.is_dir():
        return f"\n\nДіагностика: WARWATCH_MODEL_DIR={raw!r} — каталог не існує або недоступний.\n  Перевірте шлях (краще forward slashes: D:/models/warwatch)."
    pk = list(p.glob("*.pkl"))
    if not pk:
        return (
            f"\n\nДіагностика: у {p.resolve()} немає жодного *.pkl.\n"
            "  Переконайтесь, що файл моделі саме з розширенням .pkl лежить у цій папці."
        )
    return ""


def _default_weather_json_path() -> Path:
    """Шлях до JSON погоди (з урахуванням WARWATCH_LOCAL_DATA_DIR, як у бекенді)."""
    from forecasting.local_live_features import weather_history_path

    return weather_history_path()


def _pick_date_from_weather_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    dates = sorted(
        k
        for k in data
        if isinstance(k, str) and len(k) >= 10 and k[0:4].isdigit() and k[4] == "-"
    )
    return dates[-1] if dates else None


def _run_probe_mode() -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    date = "2026-05-12"
    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            model_dir = tdp / "models"
            model_dir.mkdir()
            model_path = model_dir / "randomforest_weather_probe.pkl"

            X = np.array([[-25.0], [5.0], [40.0]], dtype=np.float64)
            y = np.array([0, 0, 1], dtype=np.int32)
            clf = LogisticRegression(random_state=42, max_iter=200)
            clf.fit(X, y)
            with open(model_path, "wb") as f:
                pickle.dump(clf, f)
            sidecar = model_path.parent / f"{model_path.name}.features.json"
            sidecar.write_text(json.dumps(["day_temp"]), encoding="utf-8")

            data_root = tdp / "local_live"
            data_root.mkdir(parents=True, exist_ok=True)

            os.environ["WARWATCH_MODEL_DIR"] = str(model_dir)
            os.environ["WARWATCH_MODEL_ALARM"] = str(model_path)
            os.environ["WARWATCH_LOCAL_DATA_DIR"] = str(data_root)
            os.environ["WARWATCH_NO_TEMPLATE_CALENDAR"] = "1"

            import forecasting.local_live_features as llf

            llf._WH_JSON_CACHE = (None, None)  # noqa: SLF001

            from forecasting.prediction_service import predict_event_probabilities

            def _write_weather(root: Path, d: str, temp: float, conditions: str) -> None:
                path = root / "weather_history.json"
                payload = {
                    d: {
                        "updated_at": "12:00:00",
                        "regions": {
                            "Kyiv": {
                                "temp": temp,
                                "humidity": 70.0,
                                "windspeed": 12.0,
                                "conditions": conditions,
                            }
                        },
                    }
                }
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            def prob_for(temp: float, cond: str) -> tuple[float, dict]:
                _write_weather(data_root, date, temp, cond)
                llf._WH_JSON_CACHE = (None, None)  # noqa: SLF001
                out = predict_event_probabilities("Kyiv", date, alarm_model=str(model_path))
                return float(out["alarm_prob"]), out.get("live_context") or {}

            p_cold, ctx_cold = prob_for(-22.0, "Overcast")
            p_hot, ctx_hot = prob_for(38.0, "Clear")

            wpath = data_root / "weather_history.json"
            wpath.unlink(missing_ok=True)
            llf._WH_JSON_CACHE = (None, None)  # noqa: SLF001
            out_none = predict_event_probabilities(
                "Kyiv", date, alarm_model=str(model_path)
            )
            p_none = float(out_none["alarm_prob"])
            ctx_none = out_none.get("live_context") or {}

            print("--- PROBE (синтетична модель + тимчасовий JSON) ---")
            print(f"alarm_prob cold: {p_cold:.6f} | hot: {p_hot:.6f} | no file: {p_none:.6f}")
            print("live_context.weather (cold):", ctx_cold.get("weather"))
            print("live_context.weather (hot): ", ctx_hot.get("weather"))

            ok_meta = (
                isinstance(ctx_cold.get("weather"), dict)
                and ctx_cold["weather"].get("weather_city") == "Kyiv"
            )
            ok_diff = abs(p_hot - p_cold) > 1e-6
            ok_none = not ctx_none.get("weather")

            if not ok_meta:
                raise SystemExit("FAIL: live_context.weather missing or wrong city")
            if not ok_diff:
                raise SystemExit("FAIL: hot and cold produced identical alarm_prob")
            if not ok_none:
                raise SystemExit(f"FAIL: expected no weather context, got {ctx_none.get('weather')!r}")

            print("OK (probe): synthetic model reacts to weather overrides.")
            print("OK (probe): live_context.weather when file present.")
    finally:
        for k in (
            "WARWATCH_MODEL_DIR",
            "WARWATCH_MODEL_ALARM",
            "WARWATCH_LOCAL_DATA_DIR",
            "WARWATCH_NO_TEMPLATE_CALENDAR",
        ):
            os.environ.pop(k, None)


def _run_real_mode(region: str, date: str | None) -> None:
    import forecasting.local_live_features as llf

    from forecasting.paths import model_dir
    from forecasting.prediction_service import predict_event_probabilities

    wpath = _default_weather_json_path()
    picked_date = date or _pick_date_from_weather_file(wpath) or datetime.now().strftime("%Y-%m-%d")

    mdir = model_dir()
    pkls = sorted(mdir.glob("*.pkl")) if mdir.is_dir() else []
    if not pkls:
        diag = _model_dir_diagnostics()
        env_hint = ""
        if not (_REPO_ROOT / ".env").is_file():
            env_hint = (
                "\n\nФайл .env у корені репозиторію не знайдено.\n"
                "  Створіть .env (наприклад, скопіюйте .env.example → .env) і додайте рядок:\n"
                "    WARWATCH_MODEL_DIR=D:/шлях/до/каталогу_з_pkl\n"
            )
        raise SystemExit(
            f"Немає .pkl у {mdir.resolve()}.\n\n"
            "Варіант B — каталог із вашою моделлю:\n"
            "  PowerShell (одна сесія):\n"
            '    $env:WARWATCH_MODEL_DIR = "D:\\models\\warwatch"\n'
            "    python forecasting\\verify_weather_integration.py\n"
            "  або додайте в .env у корені репозиторію (рядок без зовнішніх лапок):\n"
            "    WARWATCH_MODEL_DIR=D:/models/warwatch\n"
            "  (змінні в оболонці мають пріоритет над .env.)\n\n"
            "Або покладіть .pkl у папки models/ або forecasting/ у репо.\n"
            "Синтетична перевірка без .pkl: --probe"
            f"{diag}{env_hint}"
        )

    # Як у проді: не встановлюємо WARWATCH_LOCAL_DATA_DIR → data/local_live/
    os.environ.pop("WARWATCH_DISABLE_WEATHER_HISTORY", None)
    llf._WH_JSON_CACHE = (None, None)  # noqa: SLF001

    print("--- РЕАЛЬНИЙ прогноз (ваша модель + data/local_live/weather_history.json) ---")
    print(f"model_dir: {mdir.resolve()}")
    print(f"alarm .pkl: {pkls[0].name} (+ інші, якщо є; обрання як у API)")
    print(f"weather_history.json: {wpath} (існує: {wpath.is_file()})")
    print(f"region={region!r}, date={picked_date!r}")
    print()

    out_with: dict
    try:
        out_with = predict_event_probabilities(region, picked_date, alarm_model=None)
    except Exception as e:
        raise SystemExit(f"Помилка інференсу (з погодою): {e}") from e

    p_with = float(out_with["alarm_prob"])
    ctx_with = out_with.get("live_context") or {}
    wmeta = ctx_with.get("weather")

    os.environ["WARWATCH_DISABLE_WEATHER_HISTORY"] = "1"
    llf._WH_JSON_CACHE = (None, None)  # noqa: SLF001

    try:
        out_without = predict_event_probabilities(region, picked_date, alarm_model=None)
    except Exception as e:
        os.environ.pop("WARWATCH_DISABLE_WEATHER_HISTORY", None)
        raise SystemExit(f"Помилка інференсу (без погоди): {e}") from e

    os.environ.pop("WARWATCH_DISABLE_WEATHER_HISTORY", None)

    p_without = float(out_without["alarm_prob"])
    ctx_without = out_without.get("live_context") or {}

    print(f"alarm_prob (з погодою з файлу):     {p_with:.6f}")
    print(f"alarm_prob (WARWATCH_DISABLE_WEATHER_HISTORY=1): {p_without:.6f}")
    print(f"різница (з − без):                  {p_with - p_without:+.6f}")
    print()
    print("models:", out_with.get("models"))
    print("live_context.weather:", wmeta)
    print()
    if not wpath.is_file():
        print("УВАГА: файлу погоди немає — обидва прогнози можуть збігатися.")
    elif not wmeta:
        print(
            "УВАГА: live_context без weather — перевірте місто в JSON "
            f"(очікується ключ для API-регіону) або дату {picked_date!r} у файлі."
        )
    elif abs(p_with - p_without) < 1e-9:
        print(
            "УВАГА: ймовірності однакові — модель може майже не використовувати "
            "погодні колонки, або погодні значення збігаються з шаблоном після злиття."
        )
    else:
        print("OK: реальна модель дає різний alarm_prob з файлом погоди vs без нього.")


def main() -> None:
    _bootstrap_env_from_repo_dotenv()

    parser = argparse.ArgumentParser(description="Перевірка weather_history → predict")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Синтетична модель + тимчасовий JSON (без вашого .pkl)",
    )
    parser.add_argument("--region", default="Kyiv", help="Як у GET /predict, напр. Kharkiv")
    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD; за замовчуванням — остання дата з weather_history.json або сьогодні",
    )
    args = parser.parse_args()

    if args.probe:
        _run_probe_mode()
    else:
        _run_real_mode(region=args.region.strip(), date=args.date)


if __name__ == "__main__":
    main()
