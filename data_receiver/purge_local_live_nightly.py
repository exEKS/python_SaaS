"""Wait until 23:59 local time each day, then delete JSON files under local_live storage."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from forecasting.local_live_features import purge_local_live_storage


def _seconds_until_235959() -> float:
    now = datetime.now()
    target = now.replace(hour=23, minute=59, second=59, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def main() -> None:
    print("Local-live purge: runs daily at 23:59:59 (local server time). Ctrl+C to stop.")
    while True:
        wait_s = _seconds_until_235959()
        h, rem = divmod(int(wait_s), 3600)
        m, s = divmod(rem, 60)
        print(f"Next purge in {h}h {m}m {s}s …")
        time.sleep(wait_s)
        info = purge_local_live_storage()
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Purged local_live: {info}")


if __name__ == "__main__":
    main()
