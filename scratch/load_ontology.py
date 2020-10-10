from owlready2 import *

ontology = get_ontology('data/raw/chebi_lite.owl')
ontology.load()

a = [c for c in ontology.classes()]
print(len(a))
