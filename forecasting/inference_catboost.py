#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pandas as pd

from forecasting.default_feature_row import DEMO_BATCH_DICT, N_TEST
from forecasting.model_runtime import (
    align_to_estimator,
    load_pickled_estimator,
    list_model_pickles,
    unwrap_estimator,
)
from forecasting.paths import model_dir

TEST_DATA_DICT = DEMO_BATCH_DICT


def _validate_test_dict(d: dict[str, list]) -> None:
    for k, v in d.items():
        if len(v) != N_TEST:
            raise ValueError(f"Column {k!r} must have length {N_TEST}, got {len(v)}")


def run_one_pkl(path, raw_df: pd.DataFrame) -> bool:
    print(f"\n{'=' * 60}\n{path.name}\n{'=' * 60}")
    loaded = load_pickled_estimator(path)
    est = unwrap_estimator(loaded)
    print(f"  Estimator type: {type(est).__name__}")
    X = align_to_estimator(est, raw_df, silent=False)
    y_hat = np.asarray(est.predict(X)).ravel()
    out = pd.DataFrame({"row": np.arange(N_TEST), "pred": y_hat})
    if hasattr(est, "predict_proba"):
        try:
            probas = est.predict_proba(X)
            if probas.ndim == 2 and probas.shape[1] >= 2:
                out["proba_class1"] = probas[:, 1]
        except (AttributeError, TypeError, ValueError):
            pass
    print(out.to_string(index=False))
    return True


def main() -> int:
    _validate_test_dict(TEST_DATA_DICT)
    raw_df = pd.DataFrame(TEST_DATA_DICT)

    mdir = model_dir()
    print(f"Model directory: {mdir.resolve()}")
    pkls = list_model_pickles(mdir)

    ok, failed = 0, 0
    for p in pkls:
        try:
            if run_one_pkl(p, raw_df):
                ok += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}", file=sys.stderr)
            traceback.print_exc(limit=2)

    print(f"\nDone. OK: {ok}, failed: {failed}, total .pkl: {len(pkls)}")
    strict = os.environ.get("WARWATCH_STRICT_EXIT", "").lower() in ("1", "true", "yes")
    if strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
