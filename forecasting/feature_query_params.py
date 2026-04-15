from __future__ import annotations

from starlette.datastructures import QueryParams

QUERY_PARAM_TO_COLUMN: dict[str, str] = {
    "feat_alarm_roll7_mean": "alarm_roll7_mean",
    "feat_day_temp": "day_temp",
    "feat_alarm_lag1": "alarm_lag1",
    "feat_alarm_lag2": "alarm_lag2",
    "feat_text_intensity_index": "text_intensity_index",
    "feat_alarm_count": "alarm_count",
    "feat_alarm_total_duration_min": "alarm_total_duration_min",
    "feat_alarm_all_region": "alarm_all_region",
    "feat_day_tempmax": "day_tempmax",
    "feat_day_tempmin": "day_tempmin",
    "feat_day_humidity": "day_humidity",
    "feat_day_windspeed": "day_windspeed",
    "feat_reddit_post_count": "reddit_post_count",
    "feat_reddit_avg_score": "reddit_avg_score",
    "feat_duration_lag1": "duration_lag1",
}


def feature_overrides_from_query_params(qp: QueryParams) -> dict[str, float] | None:
    out: dict[str, float] = {}
    for qname, col in QUERY_PARAM_TO_COLUMN.items():
        if qname not in qp:
            continue
        raw = qp.get(qname)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            out[col] = float(raw)
        except ValueError as e:
            raise ValueError(f"Invalid float query parameter {qname}={raw!r}") from e
    return out or None


def supported_feature_query_params() -> dict[str, list[dict[str, str]]]:
    rows = [
        {"query": q, "column": c}
        for q, c in sorted(QUERY_PARAM_TO_COLUMN.items(), key=lambda x: x[0])
    ]
    return {"params": rows}
