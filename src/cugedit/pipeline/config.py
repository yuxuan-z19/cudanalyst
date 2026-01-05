import copy
from dataclasses import dataclass, field

import dacite
import yaml

from ..module.config import ModuleCfg, mask_to_cfg


@dataclass
class AnalysisCfg:
    chat_config_path: str
    debug_cfg: ModuleCfg = field(default_factory=ModuleCfg)
    anlz_cfg: ModuleCfg = field(default_factory=ModuleCfg)
    perf_cfg: ModuleCfg = field(default_factory=ModuleCfg)
    plan_cfg: ModuleCfg = field(default_factory=ModuleCfg)


@dataclass(frozen=True)
class AnalysisMask:
    debug: int = 0
    anlz: int = 0
    perf: int = 0
    plan: int = 0

    def __str__(self):
        return f"p{self.plan}-d{self.debug}-a{self.anlz}-p{self.perf}"


def load_analysis_cfg(yaml_path: str) -> AnalysisCfg:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return dacite.from_dict(data_class=AnalysisCfg, data=data)


def apply_config_mask(base_config: AnalysisCfg, mask: AnalysisMask) -> AnalysisCfg:
    config = copy.deepcopy(base_config)
    config.debug_cfg = mask_to_cfg(mask.debug)
    config.anlz_cfg = mask_to_cfg(mask.anlz)
    config.perf_cfg = mask_to_cfg(mask.perf)
    config.plan_cfg = mask_to_cfg(mask.plan)
    return config
