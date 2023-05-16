from typing import List
from collections import defaultdict, OrderedDict

import torch
import torch_geometric
from torch_geometric.data import Data

from src.language.fof import ConjunctiveFormula
from src.structure.knowledge_graph import KnowledgeGraph


class QueryGraph(Data):
    def __init__(self, input_formula: ConjunctiveFormula):
        name2node, name2edge = OrderedDict(), OrderedDict()
        node_num, edge_num = 0, 0
        all_edges, pos_edge = [], []
        for term_name in input_formula.term_dict:
            name2node[term_name] = node_num
            node_num += 1
        for pred in input_formula.predicate_dict.values():
            name2edge[pred.name] = edge_num
            edge_num += 1
            all_edges.append(torch.tensor([[name2node[pred.head.name]], [name2node[pred.tail.name]]]))
            this_edge_pos = torch.tensor([-1]) if pred.skolem_negation else torch.tensor([1])
            pos_edge.append(this_edge_pos)
            # pred_triples = (self.name2node[pred.head.name], edge_num, self.name2node[pred.tail.name])
        super().__init__(edge_index=torch.cat(all_edges, dim=-1), num_nodes=len(name2node))
        self.name2node, self.name2edge = name2node, name2edge
        self.pos_edge = torch.cat(pos_edge, dim=-1)
        self.edge_grounded_rel_id = torch.zeros([len(self.name2edge),
                                                 len(input_formula.pred_grounded_relation_id_dict['e1'])])
        for pred_name in self.name2edge:
            self.edge_grounded_rel_id[self.name2edge[pred_name]] = \
                torch.tensor(input_formula.pred_grounded_relation_id_dict[pred_name])
        self.node_grounded_entity_id_dict = input_formula.term_grounded_entity_id_dict
        self.lstr = input_formula.lstr

    def deterministic_query(self, index, skip_predicate: List = [], return_full_match: bool = False):
        """
        The skip predicate is used in grounding the predicate, it can avoid creating new instance of Formula.
        """



#class KnowledgeGraph(Data):





