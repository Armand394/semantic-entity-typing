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

# === 3. CHARGEMENT DU FICHIER ET_train.txt ET AJOUT DES ENTITÉS ===
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

# === 4. FILTRER LES ENTITÉS DE ET_test.txt ET ET_valid.txt ===
def filter_et_by_kg(et_dict, kg_dict):
    """Filtre les entités de et_dict pour ne garder que celles qui sont dans kg_dict"""
    return {entity: types for entity, types in et_dict.items() if entity in kg_dict}

# === 5. FONCTION D'EXTRACTION DU CLUSTER ===
def extract_cluster(type_str):
    """Extrait le cluster à partir du type donné en suivant les règles spécifiées"""
    parts = type_str.split("/")
    
    if type_str.startswith("/base/") and len(parts) > 2:
        return f"/base/{parts[2]}"  # Cluster = /base/deuxième mot
    elif len(parts) > 1:
        return parts[1]  # Cluster = premier mot du type
    else:
        return "unknown"

# === 6. CONSTRUCTION DU FICHIER FINAL ===
def construct_output(kg_dict, et_train_dict, extra_et_dict, et_test_dict, entite_dict, relation_dict, type_dict, output_file, mode="train"):
    """Construit le fichier texte final avec la structure demandée, incluant les entités supplémentaires"""
    if mode == "train":
        # Inclure toutes les relations du KG et des entités (train + valid)
        all_entities = set(kg_dict.keys()).union(set(et_train_dict.keys())).union(set(extra_et_dict.keys()))
    elif mode == "test":
        # Ne garder que les entités présentes dans et_test_dict
        all_entities = set(et_test_dict.keys())
    else:
        raise ValueError("Mode non valide. Utilisez 'train' ou 'test'.")

    with open(output_file, "w", encoding="utf-8") as f:
        for entity in all_entities:
            if mode == "test" and entity not in et_test_dict:
                continue  # Ne pas inclure des entités qui ne sont pas dans le test

            entity_name = entite_dict.get(entity, entity)  # Récupération du nom de l'entité

            # Partie des types et clusters
            types = et_train_dict.get(entity, []) + extra_et_dict.get(entity, [])
            types_part = " [SEP] ".join([f"{entity_name} {extract_cluster(type_dict.get(t, t))} {type_dict.get(t, t)}" for t in types])

            # Partie des relations (avec les relations inverses)
            relations_part = " [SEP] ".join([f"{entity_name} {relation_dict.get(rel, rel)} {entite_dict.get(tail, tail)}"
                                            for rel, tail in kg_dict.get(entity, [])])

            # Écriture dans le fichier
            f.write(f"{entity_name} ||| {types_part} ||| {relations_part} ||| cluster \n")

# === 7. EXÉCUTION DU CODE ===
# Chargement des dictionnaires à partir des fichiers TSV
entite_dict = load_tsv("entities.tsv")
relation_dict = load_tsv("relations.tsv")
type_dict = load_tsv("types.tsv")

# Chargement des données KG et ET
kg_dict = load_kg("KG_train.txt")
et_train_dict = load_et("ET_train.txt")
et_valid_dict = load_et("ET_valid.txt")

# Fichier train (incluant ET_train et ET_valid)
construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, "LMET_train2.txt", mode="train")

# Fichier test (uniquement avec les entités dans ET_test)
et_test_dict = load_et("ET_test.txt")
construct_output(kg_dict, et_train_dict, {}, et_test_dict, entite_dict, relation_dict, type_dict, "LMET_test2.txt", mode="test")

# Fichier valid (en utilisant les mêmes relations que pour train et test, mais pour les entités de valid)
construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, "LMET_valid2.txt", mode="train")

print("Fichiers générés avec succès ! 🚀")
