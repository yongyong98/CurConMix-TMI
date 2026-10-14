import ast
import os
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


def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return _to_plain_value(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        pass
    plain_value = _to_plain_value(cfg)
    return plain_value if isinstance(plain_value, dict) else {}


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
    def __init__(self, run=None, enabled=False, project="", entity="", mode="disabled", name="", group="", tags=None):
        self.run = run
        self.enabled = enabled and run is not None
        self.project = project
        self.entity = entity or ""
        self.mode = mode
        self.name = name
        self.group = group or ""
        self.tags = list(tags or [])
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

    def finish(self):
        if not self.enabled:
            return
        try:
            self.run.finish()
        except Exception as exc:  # pragma: no cover - fail-open safety
            self._disable(exc)


def init_safe_wandb_from_cfg(cfg: Any, phase: str, default_project: str = "CurConMix"):
    if not getattr(cfg, "wandb_log", False):
        return SafeWandbRun.disabled()
    if wandb is None:
        print(f"[wandb] unavailable, continuing without logging: {_WANDB_IMPORT_ERROR}")
        return SafeWandbRun.disabled()

    config_dict = _cfg_to_dict(cfg)
    split_selector = str(getattr(cfg, "split_selector", "") or getattr(cfg, "challenge_split", "") or "dataset")
    project = str(getattr(cfg, "wandb_project", default_project) or default_project)
    entity = str(getattr(cfg, "wandb_entity", "") or "")
    group = str(getattr(cfg, "wandb_group", "") or "")
    mode = _choose_mode(getattr(cfg, "wandb_mode", "auto"))
    if mode == "disabled":
        return SafeWandbRun.disabled()

    os.environ["WANDB_CONSOLE"] = "off"
    wandb_dir = str(getattr(cfg, "wandb_dir", "") or "").strip() or None
    tags = _normalize_tags(
        getattr(cfg, "wandb_tags", []),
        default_tags=[split_selector, str(getattr(cfg, "exp", "run")), str(phase)],
    )
    model_token = str(getattr(cfg, "model_name", "model")).replace("/", "_").replace("\\", "_")
    run_name = f"{split_selector}_{model_token}_{phase}_{getattr(cfg, 'exp', 'run')}"

    init_kwargs = {
        "project": project,
        "name": run_name,
        "group": group or None,
        "tags": tags,
        "mode": mode,
        "dir": wandb_dir,
        "config": config_dict,
        "reinit": True,
    }
    if entity:
        init_kwargs["entity"] = entity

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        if mode == "online":
            print(f"[wandb] online init failed, retrying offline: {exc}")
            try:
                init_kwargs["mode"] = "offline"
                run = wandb.init(**init_kwargs)
                return SafeWandbRun(run=run, enabled=True, project=project, entity=entity, mode="offline", name=run_name, group=group, tags=tags)
            except Exception as retry_exc:
                print(f"[wandb] offline retry failed, continuing without logging: {retry_exc}")
                return SafeWandbRun.disabled()
        print(f"[wandb] init failed, continuing without logging: {exc}")
        return SafeWandbRun.disabled()

    return SafeWandbRun(run=run, enabled=True, project=project, entity=entity, mode=mode, name=run_name, group=group, tags=tags)
