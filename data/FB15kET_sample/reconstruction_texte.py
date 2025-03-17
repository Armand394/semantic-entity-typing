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

# === 2. CHARGEMENT DU FICHIER KG_train.txt AVEC RELATIONS INVERSÉES ===
def load_kg(file_path):
    """Charge les relations sous forme de dictionnaire {entité: [(relation, objet)]} en ajoutant les relations inverses"""
    kg = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                head, relation, tail = parts
                
                # Ajouter la relation originale
                if head not in kg:
                    kg[head] = []
                kg[head].append((relation, tail))

                # Ajouter la relation inverse
                inverse_relation = "inv-" + relation
                if tail not in kg:
                    kg[tail] = []
                kg[tail].append((inverse_relation, head))
    return kg

# === 3. CHARGEMENT DU FICHIER ET_train.txt ===
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

# === 4. FONCTION D'EXTRACTION DU CLUSTER ===
def extract_cluster(type_str):
    """Extrait le cluster à partir du type donné en suivant les règles spécifiées"""
    parts = type_str.split("/")
    
    if type_str.startswith("/base/") and len(parts) > 1:
        return f"/base/{parts[2]}"  # Cluster = /base/deuxième mot
    elif len(parts) > 0:
        return parts[1]  # Cluster = premier mot du type
    else:
        return "unknown"

# === 5. CONSTRUCTION DU FICHIER FINAL ===
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

            # Partie des relations (avec les relations inverses)
            relations_part = " [SEP] ".join([
                f"{entity_name} {relation_dict.get(rel, rel)} {entite_dict.get(tail, tail)}"
                for rel, tail in kg_dict.get(entity, [])
            ])

            # Écriture dans le fichier
            f.write(f"{entity_name} ||| {types_part} ||| {relations_part} ||| cluster \n")

# === 6. EXÉCUTION DU CODE ===
# Charger les fichiers TSV avec la correction
entite_dict = load_tsv("entities.tsv")
relation_dict = load_tsv("relations.tsv")
type_dict = load_tsv("types.tsv")

# Charger les fichiers de relations et de types avec ajout des relations inverses
kg_dict = load_kg("KG_train.txt")
et_dict = load_et("ET_train.txt")

# Générer le fichier de sortie
construct_output(kg_dict, et_dict, entite_dict, relation_dict, type_dict, "LMET_train2.txt")

kg_dict = load_kg("KG_train.txt")
et_dict = load_et("ET_test.txt")

# Générer le fichier de sortie
construct_output(kg_dict, et_dict, entite_dict, relation_dict, type_dict, "LMET_test2.txt")

kg_dict = load_kg("KG_train.txt")
et_dict = load_et("ET_valid.txt")

# Générer le fichier de sortie
construct_output(kg_dict, et_dict, entite_dict, relation_dict, type_dict, "LMET_valid2.txt")

