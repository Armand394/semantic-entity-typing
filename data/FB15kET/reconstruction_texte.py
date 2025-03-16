import os

# === 1. CHARGEMENT DES FICHIERS TSV ===
def load_tsv(file_path):
    """Charge un fichier TSV sous forme de dictionnaire {id: nom}"""
    data_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                data_dict[parts[1]] = parts[0]  # Correction : on mappe l'ID vers le nom
    return data_dict

# === 2. CHARGEMENT DES FICHIERS KG_train.txt et ET_train.txt ===
def load_kg(file_path):
    """Charge les relations sous forme de dictionnaire {entité: [(relation, objet)]}"""
    kg = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                head, relation, tail = parts
                if head not in kg:
                    kg[head] = []
                kg[head].append((relation, tail))
    return kg

def load_et(file_path):
    """Charge les types sous forme de dictionnaire {entité: [types]}"""
    et = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity, entity_type = parts
                if entity not in et:
                    et[entity] = []
                et[entity].append(entity_type)
    return et

# === 3. FONCTION D'EXTRACTION DU CLUSTER ===
def extract_cluster(type_str):
    """Extrait le cluster à partir du type donné en suivant les règles spécifiées"""
    parts = type_str.split("/")
    
    if type_str.startswith("/base/") and len(parts) > 1:
        return f"/base/{parts[2]}"  # Cluster = /base/deuxième mot
    elif len(parts) > 0:
        return parts[1]  # Cluster = premier mot du type
    else:
        return "unknown"

# === 4. CONSTRUCTION DU FICHIER FINAL ===
def construct_output(kg_dict, et_dict, entite_dict, relation_dict, type_dict, output_file):
    """Construit le fichier texte final avec la structure demandée"""
    with open(output_file, "w", encoding="utf-8") as f:
        for entity in kg_dict.keys():  # On parcourt les entités connues dans KG_train
            entity_name = entite_dict.get(entity, entity)  # Récupération du nom de l'entité

            # Partie des types et clusters
            types_part = " [SEP] ".join([
                f"{entity_name} {extract_cluster(type_dict.get(t, t))} {type_dict.get(t, t)}"
                for t in et_dict.get(entity, [])
            ])

            # Partie des relations
            relations_part = " [SEP] ".join([
                f"{entity_name} {relation_dict.get(rel, rel)} {entite_dict.get(tail, tail)}"
                for rel, tail in kg_dict.get(entity, [])
            ])

            # Écriture dans le fichier
            f.write(f"{entity_name} ||| {types_part} ||| {relations_part} ||| cluster \n")

# === 5. EXÉCUTION DU CODE ===
# Charger les fichiers TSV avec la correction
entite_dict = load_tsv("entities.tsv")
relation_dict = load_tsv("relations.tsv")
type_dict = load_tsv("types.tsv")

# Charger les fichiers de relations et de types
kg_dict = load_kg("KG_train.txt")
et_dict = load_et("ET_train.txt")

# Générer le fichier de sortie
construct_output(kg_dict, et_dict, entite_dict, relation_dict, type_dict, "output.txt")
