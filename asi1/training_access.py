from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path

from .session_store import SessionStore


def update_env(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(values)
    updated: list[str] = []
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            updated.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            updated.append(f"{key}={remaining.pop(key)}")
        else:
            updated.append(line)
    if updated and updated[-1] != "":
        updated.append("")
    updated.extend(f"{key}={value}" for key, value in remaining.items())
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    path.chmod(0o600)


def write_access_file(path: Path, *, expires_at: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"enabled": True, "expires_at": expires_at}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def enable(env_path: Path, *, ttl_seconds: int, access_path: Path) -> tuple[str, int]:
    if ttl_seconds < 60 or ttl_seconds > 3600:
        raise ValueError("TTL must be between 60 and 3600 seconds")
    code = secrets.token_urlsafe(18)
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    expires_at = int(time.time()) + ttl_seconds
    update_env(
        env_path,
        {
            "DAPTA_ASI1_ENABLE_FACTORY_BUILDS": "true",
            "DAPTA_ASI1_FACTORY_BUILD_CODE_SHA256": digest,
            "DAPTA_ASI1_FACTORY_BUILD_EXPIRES_AT": str(expires_at),
        },
    )
    write_access_file(access_path, expires_at=expires_at)
    return code, expires_at


def disable(env_path: Path, state_path: Path, access_path: Path) -> int:
    update_env(
        env_path,
        {
            "DAPTA_ASI1_ENABLE_FACTORY_BUILDS": "false",
            "DAPTA_ASI1_FACTORY_BUILD_CODE_SHA256": "",
            "DAPTA_ASI1_FACTORY_BUILD_EXPIRES_AT": "",
        },
    )
    access_path.unlink(missing_ok=True)
    return SessionStore(state_path).clear_pending_builds()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage one-shot ASI factory access")
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--access-file", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    enable_parser = subparsers.add_parser("enable")
    enable_parser.add_argument("--ttl-seconds", type=int, default=600)
    subparsers.add_parser("disable")
    args = parser.parse_args()
    if args.action == "enable":
        code, expires_at = enable(
            args.env, ttl_seconds=args.ttl_seconds, access_path=args.access_file
        )
        print(f"ACCESS_CODE={code}")
        print(f"EXPIRES_AT={expires_at}")
        print("SMOKE_LIMITS=10 examples, 300 training seconds, 1 round")
    else:
        cleared = disable(args.env, args.state, args.access_file)
        print(f"CLEARED_PENDING_BUILDS={cleared}")
        print("TRAINING_ACCESS=disabled")


if __name__ == "__main__":
    main()
