from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_url(url: str, timeout: float) -> dict:
    rec = {"url": url, "checked_utc": utc_now_iso(), "ok": False}
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            rec.update(
                {
                    "ok": 200 <= int(r.status) < 300,
                    "status": int(r.status),
                    "content_type": r.headers.get("Content-Type", ""),
                    "content_length": r.headers.get("Content-Length", ""),
                }
            )
            return rec
    except urllib.error.HTTPError as e:
        details = ""
        try:
            details = e.read().decode("utf-8", "ignore")[:1000]
        except Exception:
            details = ""

        if not details:
            # Some endpoints return empty body for HEAD; retry lightweight GET for diagnostics.
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r2:  # pragma: no cover
                    details = f"GET succeeded unexpectedly: status={int(r2.status)}"
            except urllib.error.HTTPError as e2:
                try:
                    details = e2.read().decode("utf-8", "ignore")[:1000]
                except Exception:
                    details = details
            except Exception:
                pass

        rec.update(
            {
                "status": int(getattr(e, "code", 0) or 0),
                "error": f"HTTPError: {e}",
                "details": details,
            }
        )
        return rec
    except Exception as e:  # pragma: no cover
        rec.update({"status": 0, "error": f"{type(e).__name__}: {e}", "details": ""})
        return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check connectivity to official LRA source endpoints.")
    p.add_argument("--out", default="output/official_source_connectivity.json")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--strict", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    urls = [
        "https://storage.googleapis.com/long-range-arena/lra_release",
        "https://storage.googleapis.com/long-range-arena/lra_release.gz",
    ]

    checks = [check_url(u, timeout=float(args.timeout)) for u in urls]
    ok_any = any(bool(c.get("ok")) for c in checks)

    report = {
        "checked_utc": utc_now_iso(),
        "ok_any": ok_any,
        "checks": checks,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok_any": ok_any, "out": str(out)}, ensure_ascii=False, indent=2))

    if args.strict and not ok_any:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
