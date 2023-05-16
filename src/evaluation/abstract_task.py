from abc import abstractmethod
from typing import Dict


class AbstractTask:
    @abstractmethod
    def evaluate_nbp(self, nbp) -> Dict:
        pass