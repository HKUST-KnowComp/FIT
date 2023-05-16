from .abstract_task import AbstractTask
from typing import Dict
import json

from torch.utils.data import DataLoader

# from ..utils.data import collate_qaa_into_first_order_formula

class QueryAnsweringTV(AbstractTask):
    def __init__(self, qaafile, **dataloader_kwargs) -> None:
        with open(qaafile, 'rt') as f:
            self.lstr_qaa = json.load(f)

        self.lstr_dataloader = {}
        self.dataloader_kwargs = dataloader_kwargs

    # TODO evaluation task
    def evaluate_nbp(self, nbp) -> Dict:
        return super().evaluate_nbp(nbp)