"""Strategy registry: expand configs/strategies.yaml into 300 Strategy instances.

Each family module exposes a ``make(params) -> callable`` factory (or a Strategy
subclass) registered in ``FAMILY_FACTORIES``. The YAML declares, per family, a
parameter grid (explicit list of dicts and/or a cartesian ``grid`` spec). The
registry expands every family, builds a deterministic, filesystem-safe id, and
returns the instances in a stable order.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from ..core import Strategy, clean_position

GenerateFn = Callable[[pd.DataFrame], pd.Series]
FactoryFn = Callable[[dict[str, Any]], GenerateFn]

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "strategies.yaml"
)


class FunctionStrategy(Strategy):
    """Concrete Strategy wrapping a family factory's generate callable."""

    def __init__(
        self,
        id: str,
        family: str,
        params: dict[str, Any],
        fn: GenerateFn,
    ) -> None:
        super().__init__(id=id, family=family, params=params)
        self._fn = fn

    def generate(self, df: pd.DataFrame) -> pd.Series:
        pos = self._fn(df)
        pos = clean_position(pos, df.index)
        pos.name = self.id
        return pos


def _family_factories() -> dict[str, FactoryFn]:
    """Lazily import family modules and collect their ``make`` factories."""
    from .families import (
        breakout,
        combo,
        meanrev,
        momentum,
        oscillator,
        pattern,
        seasonal,
        trend,
    )

    modules = [
        trend,
        breakout,
        meanrev,
        momentum,
        oscillator,
        pattern,
        seasonal,
        combo,
    ]
    factories: dict[str, FactoryFn] = {}
    for mod in modules:
        registry: dict[str, FactoryFn] = getattr(mod, "FACTORIES", {})
        for key, fn in registry.items():
            if key in factories:
                raise ValueError(f"duplicate family factory key: {key}")
            factories[key] = fn
    return factories


def _fmt(value: Any) -> str:
    """Filesystem-safe token for a parameter value."""
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).replace(".", "p").replace("-", "m")
    if isinstance(value, (list, tuple)):
        return "-".join(_fmt(v) for v in value)
    return str(value).replace(".", "p").replace("-", "m")


def _expand_grid(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one family's grid spec into an ordered list of param dicts.

    Supported forms:
      * ``params:`` an explicit list of dicts (used verbatim, in order).
      * ``grid:`` a dict of name -> list; cartesian product (sorted key order).
    A ``defaults`` dict (optional) is merged under every produced combination.
    """
    defaults: dict[str, Any] = dict(spec.get("defaults", {}))
    out: list[dict[str, Any]] = []

    if "params" in spec and spec["params"]:
        for entry in spec["params"]:
            merged = dict(defaults)
            merged.update(entry)
            out.append(merged)

    if "grid" in spec and spec["grid"]:
        grid = spec["grid"]
        keys = list(grid.keys())  # preserve yaml declaration order
        value_lists = [grid[k] for k in keys]
        for combo in itertools.product(*value_lists):
            merged = dict(defaults)
            merged.update(dict(zip(keys, combo)))
            out.append(merged)

    if not out:
        out.append(dict(defaults))
    return out


def _id_for(prefix: str, params: dict[str, Any], id_keys: list[str] | None) -> str:
    """Build a deterministic id from prefix + selected param tokens."""
    keys = id_keys if id_keys is not None else list(params.keys())
    tokens = [prefix]
    for key in keys:
        if key not in params:
            continue
        tokens.append(f"{_fmt(params[key])}")
    return "_".join(tokens)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the raw strategies YAML mapping."""
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_all(path: str | Path | None = None) -> list[Strategy]:
    """Expand the YAML into the full, ordered list of Strategy instances."""
    cfg = load_config(path)
    factories = _family_factories()
    families_cfg: dict[str, Any] = cfg["families"]

    strategies: list[Strategy] = []
    seen: set[str] = set()

    for family_key, spec in families_cfg.items():
        factory_key = spec.get("factory", family_key)
        if factory_key not in factories:
            raise KeyError(
                f"family '{family_key}' references unknown factory "
                f"'{factory_key}'"
            )
        factory = factories[factory_key]
        prefix = spec.get("prefix", family_key)
        id_keys = spec.get("id_keys")
        combos = _expand_grid(spec)

        for params in combos:
            sid = _id_for(prefix, params, id_keys)
            if sid in seen:
                raise ValueError(
                    f"duplicate strategy id '{sid}' in family '{family_key}'"
                )
            seen.add(sid)
            fn = factory(params)
            strategies.append(
                FunctionStrategy(
                    id=sid, family=family_key, params=dict(params), fn=fn
                )
            )

    return strategies
