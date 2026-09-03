import json
from collections import Counter
kg=json.load(open('data/processed/kg.json'))
m=[n for n in kg['nodes'] if n.get('name')=='mace_energy_per_atom']
print(len(m))