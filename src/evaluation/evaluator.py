import logging

from ..structure.knowledge_graph_index import KGIndex

from .link_prediction import LinkPrediction
from .query_answering import QueryAnsweringTV
from ..structure import KnowledgeGraph, NeuralBinaryPredicate
from ..utils.config import EvaluationConfig
from ..utils.recorder import EvalRecorder

class Evaluator:
    def __init__(self,
                 eval_every_step,
                 eval_every_epoch,
                 logdir,
                 task_dict,
                 device,
                 observed_kg: KnowledgeGraph,
                 kgindex: KGIndex,
                 **kwargs) -> None:
        self.eval_every_step = eval_every_step
        self.eval_every_epoch = eval_every_epoch

        self.task = {}
        self.task_recorder = {}

        for k, v in task_dict.items():
            logging.info(f"initialize task {k}")
            name = v['name']
            params = v['params']

            if name.lower() == 'linkprediction':
                Task = LinkPrediction
            elif name.lower() == 'QueryAnswering':
                Task = QueryAnsweringTV
            else:
                raise NotImplementedError

            logging.info(f"\ttask type {name}: {params}")
            self.task[k] = Task.create(
                observed_kg=observed_kg,
                kgindex=kgindex,
                device=device,
                **params)
            self.task_recorder[k] = EvalRecorder(logdir, k)
            logging.info(f"task {k} initialized")

        self.dev_key = kwargs.get('dev_key', None)
        print(kwargs)
        print(self.dev_key)

    @classmethod
    def create(cls, eval_config: EvaluationConfig, logdir):
        logging.info("initalize evaluator")
        logging.info(eval_config.to_dict())

        kgindex = KGIndex.load(filename=eval_config.kgindex_file)
        observed_kg = KnowledgeGraph.create(
            triple_files=eval_config.observed_triple_filelist,
            kgindex=kgindex)

        print(eval_config.to_dict())
        return cls(observed_kg=observed_kg,
                   kgindex=kgindex,
                   logdir=logdir,
                   **eval_config.to_dict())

    def evaluate_nbp(self, nbp: NeuralBinaryPredicate, global_step, global_epoch, get_key_metric):
        for k in self.task:
            _metric = self.task[k].evaluate_nbp(nbp, prefix=k)
            metric = {
                f"{k}:{_k}" : _v for _k, _v in _metric.items()
            }
            metric['global_step'] = global_step
            metric['global_epoch'] = global_epoch
            self.task_recorder[k].write(metric)

            if get_key_metric:
                return metric["dev:filter/agg/mrr"]
