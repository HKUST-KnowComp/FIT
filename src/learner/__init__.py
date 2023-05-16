from .abstract_learner import Learner, LearnerForwardOutput
from .isomorphic import IsomorphicLearner
from .elementary import ElementaryLearner
from .subgraph import SubgraphLearner

def get(name):
    if name == 'I':
        return IsomorphicLearner
    elif name == 'E':
        return ElementaryLearner
    elif name == 'S':
        return SubgraphLearner
    else:
        raise NotImplementedError