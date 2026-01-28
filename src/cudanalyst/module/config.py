from dataclasses import dataclass
from enum import IntFlag


@dataclass
class ModuleCfg:
    enabled: bool = False
    formatted: bool = False
    summarized: bool = False


class ModuleBits(IntFlag):
    # ? b'[summarized][formatted][enabled]
    SUMMARY = 1 << 2
    FORMAT = 1 << 1
    ENABLE = 1 << 0

    MODE_FULL = ENABLE | FORMAT | SUMMARY
    MODE_RAW = ENABLE | FORMAT
    MODE_NONE = 0


def mask_to_cfg(mask: int) -> ModuleCfg:
    return ModuleCfg(
        bool(mask & ModuleBits.ENABLE),
        bool(mask & ModuleBits.FORMAT),
        bool(mask & ModuleBits.SUMMARY),
    )
