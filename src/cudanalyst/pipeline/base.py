from abc import ABC, abstractmethod

from ..module.chat import ChatConfig
from .config import AnalysisCfg


class Pipeline(ABC):
    def __init__(self, config: AnalysisCfg):
        self.config = config
        self.service = ChatConfig(config.chat_config_path)[0]

    @abstractmethod
    def run(self):
        pass
