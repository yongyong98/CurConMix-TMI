import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import wandb
except Exception as exc:  # pragma: no cover - optional dependency safety
    wandb = None
    _WANDB_IMPORT_ERROR = exc
else:
    _WANDB_IMPORT_ERROR = None


def _to_plain_value(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(k): _to_plain_value(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _normalize_tags(tags: Any, default_tags: Iterable[str]):
    parsed = tags
    if isinstance(tags, str):
        stripped = tags.strip()
        if not stripped:
            parsed = []
        else:
            try:
                literal = ast.literal_eval(stripped)
            except Exception:
                parsed = [part.strip() for part in stripped.split(",") if part.strip()]
            else:
                parsed = literal
    if parsed is None:
        parsed = []
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    ordered = []
    seen = set()
    for tag in list(parsed) + list(default_tags):
        if tag is None:
            continue
        tag_str = str(tag).strip()
        if not tag_str or tag_str in seen:
            continue
        ordered.append(tag_str)
        seen.add(tag_str)
    return ordered


def _choose_mode(requested_mode: str):
    requested = (requested_mode or "auto").strip().lower()
    if requested not in {"auto", "online", "offline", "disabled"}:
        requested = "auto"
    if requested == "disabled":
        return "disabled"
    if requested == "auto":
        return "online" if os.environ.get("WANDB_API_KEY") else "offline"
    return requested


class SafeWandbRun:
    def __init__(self, run=None, enabled=False, project="", entity="", mode="disabled", name="", group="", tags=None, wandb_dir=""):
        self.run = run
        self.enabled = enabled and run is not None
        self.project = project
        self.entity = entity or ""
        self.mode = mode
        self.name = name
        self.group = group or ""
        self.tags = list(tags or [])
        self.wandb_dir = wandb_dir or ""
        self.run_id = getattr(run, "id", "") if run is not None else ""
        self._warned = False

    @classmethod
    def disabled(cls):
        return cls()

    def _disable(self, exc: Exception):
        if not self._warned:
            print(f"[wandb] disabling logging after runtime error: {exc}")
            self._warned = True
        self.enabled = False
        self.run = None

    def log(self, payload: Dict[str, Any], step=None, commit=None):
        if not self.enabled or not payload:
            return
        try:
            kwargs = {}
            if step is not None:
                kwargs["step"] = step
            if commit is not None:
                kwargs["commit"] = commit
            self.run.log(_to_plain_value(payload), **kwargs)
        except Exception as exc:  # pragma: no cover - fail-open safety
            self._disable(exc)

    def update_summary(self, payload: Dict[str, Any]):
        if not self.enabled or not payload:
            return
        try:
            self.run.summary.update(_to_plain_value(payload))
        except Exception as exc:  # pragma: no cover - fail-open safety
            self._disable(exc)

    def write_session(self, session_path):
        payload = {
            "enabled": bool(self.enabled),
            "project": self.project,
            "entity": self.entity,
            "mode": self.mode,
            "name": self.name,
            "group": self.group,
            "tags": self.tags,
            "run_id": self.run_id,
            "wandb_dir": self.wandb_dir,
        }
        path = Path(session_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def finish(self):
        if not self.enabled:
            return
        try:
            self.run.finish()
        except Exception as exc:  # pragma: no cover - fail-open safety
            self._disable(exc)


def init_safe_wandb(config_dict: Dict[str, Any], *, enabled: bool, project: str, entity: str, mode: str, group: str, tags, run_name: str, wandb_dir: str):
    if not enabled:
        return SafeWandbRun.disabled()
    if wandb is None:
        print(f"[wandb] unavailable, continuing without logging: {_WANDB_IMPORT_ERROR}")
        return SafeWandbRun.disabled()

    chosen_mode = _choose_mode(mode)
    if chosen_mode == "disabled":
        return SafeWandbRun.disabled()

    os.environ["WANDB_CONSOLE"] = "off"
    normalized_tags = _normalize_tags(tags, default_tags=["mrtt"])
    init_kwargs = {
        "project": project or "CurConMix-MRTT",
        "name": run_name,
        "group": group or None,
        "tags": normalized_tags,
        "mode": chosen_mode,
        "dir": wandb_dir or None,
        "config": _to_plain_value(config_dict),
        "reinit": True,
    }
    if entity:
        init_kwargs["entity"] = entity

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        if chosen_mode == "online":
            print(f"[wandb] online init failed, retrying offline: {exc}")
            try:
                init_kwargs["mode"] = "offline"
                run = wandb.init(**init_kwargs)
                return SafeWandbRun(run=run, enabled=True, project=project, entity=entity, mode="offline", name=run_name, group=group, tags=normalized_tags, wandb_dir=wandb_dir)
            except Exception as retry_exc:
                print(f"[wandb] offline retry failed, continuing without logging: {retry_exc}")
                return SafeWandbRun.disabled()
        print(f"[wandb] init failed, continuing without logging: {exc}")
        return SafeWandbRun.disabled()

    return SafeWandbRun(run=run, enabled=True, project=project, entity=entity, mode=chosen_mode, name=run_name, group=group, tags=normalized_tags, wandb_dir=wandb_dir)


def resume_safe_wandb_from_session(session_path, extra_summary=None):
    path = Path(session_path)
    if not path.exists():
        return SafeWandbRun.disabled()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[wandb] failed to read session file, continuing without logging: {exc}")
        return SafeWandbRun.disabled()
    if not payload.get("enabled") or not payload.get("run_id"):
        return SafeWandbRun.disabled()
    if wandb is None:
        print(f"[wandb] unavailable while resuming, continuing without logging: {_WANDB_IMPORT_ERROR}")
        return SafeWandbRun.disabled()

    os.environ["WANDB_CONSOLE"] = "off"
    init_kwargs = {
        "project": payload.get("project") or "CurConMix-MRTT",
        "entity": payload.get("entity") or None,
        "id": payload["run_id"],
        "resume": "allow",
        "mode": payload.get("mode") or "offline",
        "dir": payload.get("wandb_dir") or None,
        "reinit": True,
        "name": payload.get("name") or None,
        "group": payload.get("group") or None,
        "tags": payload.get("tags") or None,
    }
    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[wandb] failed to resume run, continuing without logging: {exc}")
        return SafeWandbRun.disabled()

    resumed = SafeWandbRun(
        run=run,
        enabled=True,
        project=payload.get("project", ""),
        entity=payload.get("entity", ""),
        mode=payload.get("mode", "offline"),
        name=payload.get("name", ""),
        group=payload.get("group", ""),
        tags=payload.get("tags", []),
        wandb_dir=payload.get("wandb_dir", ""),
    )
    if extra_summary:
        resumed.update_summary(extra_summary)
    return resumed
