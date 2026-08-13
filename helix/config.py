"""Typed configuration loaded from YAML with environment-variable overrides.

Secrets never live in the YAML: the Tushare token is read from ``HELIX_TUSHARE_TOKEN``
(or a ``.env`` file next to the project root) and validated at the point of use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class DataConfig(BaseModel):
    root: Path = Path("data")
    start_date: str = "20180101"
    end_date: str = ""
    calendar_exchange: str = "SSE"
    requests_per_minute: int = Field(400, gt=0)
    max_retries: int = Field(5, ge=0)

    @field_validator("root")
    @classmethod
    def _absolutize(cls, v: Path) -> Path:
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()


class UniverseConfig(BaseModel):
    exclude_st: bool = True
    min_list_days: int = Field(120, ge=0)
    exclude_exchanges: list[str] = Field(default_factory=lambda: ["BJ"])
    min_amount_kcny: float = 20000.0
    min_close: float = 2.0
    max_close: float = 1000.0


class LabelConfig(BaseModel):
    entry_offset: int = Field(1, ge=1)
    touch_offset: int = Field(2, ge=1)
    target_ratio: float = Field(1.08, gt=1.0)
    exclude_entry_limit_up: bool = True
    limit_price_eps: float = 0.001

    @field_validator("touch_offset")
    @classmethod
    def _touch_after_entry(cls, v: int, info) -> int:
        entry = info.data.get("entry_offset", 1)
        if v < entry:
            raise ValueError("touch_offset must be >= entry_offset")
        return v


class SplitConfig(BaseModel):
    train_days: int = Field(750, gt=0)
    valid_days: int = Field(120, gt=0)
    test_days: int = Field(120, gt=0)
    step_days: int = Field(120, gt=0)
    embargo_days: int = Field(5, ge=0)


class GPConfig(BaseModel):
    population: int = Field(400, gt=1)
    generations: int = Field(25, ge=1)
    tournament_size: int = Field(7, gt=1)
    crossover_prob: float = 0.6
    mutation_prob: float = 0.3
    init_depth: tuple[int, int] = (2, 5)
    max_depth: int = 8
    max_nodes: int = 40
    windows: list[int] = Field(default_factory=lambda: [3, 5, 10, 20, 60])
    complexity_penalty: float = 0.0015
    min_daily_samples: int = 50
    min_coverage: float = 0.4
    max_abs_corr: float = 0.7
    n_keep: int = 24
    search_max_stocks: int = 2000
    hall_of_fame: int = 60
    seed: int = 20240101


class DLConfig(BaseModel):
    seq_len: int = Field(20, gt=0)
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 4096
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    pos_weight_cap: float = 20.0
    clip_sigma: float = 4.0
    seed: int = 20240101


class BacktestConfig(BaseModel):
    """Trade-level accounting. Every rate is in basis points of notional.

    Costs are charged per side rather than as one round-trip number because stamp duty
    is sell-side only, and because it halved on 2023-08-28 -- a panel starting in 2018
    straddles that cut, so the two stamp rates are separate knobs (see
    :mod:`helix.eval.backtest`).

    ``extra="forbid"`` so that a stale key in a hand-written YAML fails loudly instead
    of being silently ignored; a cost setting that does not apply is the kind of error
    that only shows up as an unexplained number.
    """

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(4, gt=0)
    exit_rule: Literal["close", "target"] = "close"
    enable_realistic_exit: bool = False
    commission_bps: float = Field(2.5, ge=0)               # 佣金，双边
    transfer_bps: float = Field(0.1, ge=0)                 # 过户费，双边
    stamp_sell_bps: float = Field(5.0, ge=0)               # 印花税，仅卖出，2023-08-28 起
    stamp_sell_bps_before_cut: float = Field(10.0, ge=0)   # 印花税，仅卖出，2023-08-28 前
    slippage_bps: float = Field(10.0, ge=0)                # 单边，叠加在法定费率之上


class Config(BaseModel):
    data: DataConfig = DataConfig()
    universe: UniverseConfig = UniverseConfig()
    label: LabelConfig = LabelConfig()
    split: SplitConfig = SplitConfig()
    gp: GPConfig = GPConfig()
    dl: DLConfig = DLConfig()
    backtest: BacktestConfig = BacktestConfig()

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"config not found: {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cfg = cls.model_validate(raw)
        env_root = os.environ.get("HELIX_DATA_ROOT")
        if env_root:
            cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"root": Path(env_root)})})
        if cfg.split.embargo_days < cfg.label.touch_offset + 1:
            raise ValueError(
                f"split.embargo_days ({cfg.split.embargo_days}) must be at least "
                f"label.touch_offset + 1 ({cfg.label.touch_offset + 1}) to purge label leakage"
            )
        return cfg


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader; existing environment variables win."""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_tushare_token() -> str:
    load_dotenv()
    token = os.environ.get("HELIX_TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HELIX_TUSHARE_TOKEN is not set. Copy .env.example to .env and add your "
            "Tushare Pro token (https://tushare.pro/user/token)."
        )
    return token
