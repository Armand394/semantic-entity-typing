import pickle

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def group_by_entity(data):
    """Crée un dictionnaire où la clé est l'entité et la valeur est une liste des triplets associés."""
    entity_dict = {}
    for triplets1, triplets2, entity in data:
        if entity not in entity_dict:
            entity_dict[entity] = {"list1": [], "list2": []}
        entity_dict[entity]["list1"].extend(triplets1)
        entity_dict[entity]["list2"].extend(triplets2)
    return entity_dict

file1 = "./data/FB15kET/LMET_train.pkl"
file2 = "./data/FB15kET/LMET_train2.pkl"

data1 = load_pkl(file1)
data2 = load_pkl(file2)

dict1 = group_by_entity(data1)
dict2 = group_by_entity(data2)

# Exemple de comparaison des nombres de triplets pour chaque entité
for entity in dict1:
    if entity in dict2:
        len1 = len(dict1[entity]["list1"]) + len(dict1[entity]["list2"])
        len2 = len(dict2[entity]["list1"]) + len(dict2[entity]["list2"])
        if len1 != len2:
            print(f"Différence pour l'entité {entity}: {len1} vs {len2}")
