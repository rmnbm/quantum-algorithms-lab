import pickle
import os

def save_instances(instances, filename="instances_pool.pkl"):
    
    with open(filename, 'wb') as f:
        pickle.dump(instances, f)
    print(f"Pool de {len(instances)} instances sauvegardé dans {filename}")

def load_instances(filename="instances_pool.pkl"):
    
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            instances = pickle.load(f)
        print(f"Chargement de {len(instances)} instances réussi.")
        return instances
    else:
        print("Erreur : Fichier de sauvegarde introuvable.")
        return None