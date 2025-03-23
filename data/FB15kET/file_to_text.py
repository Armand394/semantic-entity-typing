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

# === 2. CHARGEMENT DU FICHIER KG_train.txt AVEC RELATIONS INVERSÉES ET FILTRAGE DES SELF-LOOPS ===
def load_kg(file_path):
    """Charge les relations sous forme de dictionnaire {entité: [(relation, objet)]}, sans self-loop"""
    kg = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                head, relation, tail = parts
                
                # Exclure les self-loops
                if head == tail:
                    continue  

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
def construct_output(kg_dict, et_train_dict, et_valid_dict, et_filter_dict, entite_dict, relation_dict, type_dict, cluster_dict, output_file, mode="train"):
    """Construit le fichier texte final avec la structure demandée, incluant les entités supplémentaires"""
    
    # Charger la relation et l'entité de l'ID 0
    relation_0 = relation_dict.get("0", "0")  # Relation correspondant à l'ID 0
    entity_0 = entite_dict.get("0", "0")  # Entité correspondant à l'ID 0
    
    # Dernier cluster et type
    last_cluster = cluster_dict.get(list(cluster_dict.keys())[-1], "unknown")
    last_type = type_dict.get(list(type_dict.keys())[-1], "unknown")
    
    if mode == "train":
        all_entities = set(kg_dict.keys()).union(set(et_train_dict.keys())).union(set(et_valid_dict.keys()))
    else:  # Pour test et valid
        all_entities = set(et_filter_dict.keys())

    with open(output_file, "w", encoding="utf-8") as f:
        for entity in all_entities:
            entity_name = entite_dict.get(entity, entity)  # Récupération du nom de l'entité

            # Partie des types et clusters (on fusionne et_train_dict et et_valid_dict)
            types = et_train_dict.get(entity, []) + et_valid_dict.get(entity, [])
            types_part = " [SEP] ".join([ 
                f"{entity_name} {extract_cluster(type_dict.get(t, t))} {type_dict.get(t, t)}"
                for t in types
            ])

            # Si aucune relation n'a été trouvée (et c'est une entité de type), on remplace par le dernier cluster et type
            if not types:
                types_part = f"{entity_name} {last_cluster} {last_type}"

            # Partie des relations (avec les relations inverses et sans self-loops)
            relations_part = " [SEP] ".join([ 
                f"{entity_name} {relation_dict.get(rel, rel)} {entite_dict.get(tail, tail)}"
                for rel, tail in kg_dict.get(entity, [])
                if entity != tail  # Suppression des self-loops
            ])

            # Si aucune relation n'a été trouvée pour les triplets `entité relation entité`, on remplace par relation "0"
            if not relations_part:
                relations_part = f"{entity_name} {relation_0} {entity_0}"

            # Écriture dans le fichier
            f.write(f"{entity_name} ||| {types_part} ||| {relations_part} ||| cluster \n")

# === 7. EXÉCUTION DU CODE ===
entite_dict = load_tsv("entities.tsv")
relation_dict = load_tsv("relations.tsv")
type_dict = load_tsv("types.tsv")
cluster_dict = load_tsv("clusters.tsv")

kg_dict = load_kg("KG_train.txt")  # Applique maintenant le filtre des self-loops
et_train_dict = load_et("ET_train.txt")
et_valid_dict = load_et("ET_valid.txt")

# Génération du fichier Train (contient ET_train + ET_valid)
construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict, "LMET_train2.txt", mode="train")

# Génération du fichier Valid (même structure que Train)
construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict, "LMET_valid2.txt", mode="train")

# Chargement et filtrage du fichier ET_test
et_test_dict = load_et("ET_test.txt")
filtered_et_test_dict = filter_et_by_kg(et_test_dict, kg_dict)

# Génération du fichier Test (contient seulement ET_test mais inclut ET_train + ET_valid pour les types)
construct_output(kg_dict, et_train_dict, et_valid_dict, filtered_et_test_dict, entite_dict, relation_dict, type_dict, cluster_dict, "LMET_test2.txt", mode="test")

print("Fichiers générés avec succès, avec les règles appliquées pour les entités et relations ! 🚀")
