import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


def new_audit_id(prefix: str = "qry") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return "%s_%s_%s" % (prefix, stamp, suffix)


def write_audit_record(path: Optional[str], record: Dict[str, Any]) -> None:
    if not path:
        return
    audit_path = Path(path).expanduser()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
