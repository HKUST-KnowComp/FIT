import pprint

from src.utils.config import ExperimentConfigCollection

ecc = ExperimentConfigCollection.from_yaml_file('config/default_config.yaml')

for comp in ecc.components:
    print(comp)
    pprint.pprint(getattr(ecc, comp + '_config').to_dict())