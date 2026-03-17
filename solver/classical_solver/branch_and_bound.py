#TODO6 LAB2
class BranchAndBound :
    
    def __init__(self, items, valeurs, poids):
        self.items = items
        self.valeurs = valeurs
        self.poids = poids
        self.nb_objets = len(valeurs)

        self.objets_tries = self.trier_items()
        
    def trier_items(self):
        ratio_decroissant = []
        
        for i in range(self.nb_objets):
            v = self.valeurs[i]
            p = self.poids[i]
            ratio = v / p

            ratio_decroissant.append({'index':i , 'items' : self.items[i], 'poids': p, 'valeur': v, 'ratio': ratio})
            
        return sorted(ratio_decroissant, key=lambda x: x['ratio'], reverse=True)
        

    def calculer_borne_sup(self, index, poids_restant, valeur_actuelle):
        
        borne = valeur_actuelle
        poids_dispo = poids_restant
        j = index

        while j < self.nb_objets and self.objets_tries[j]['poids'] <= poids_dispo:
            poids_dispo -= self.objets_tries[j]['poids']
            borne += self.objets_tries[j]['valeur']
            j += 1

        if j < self.nb_objets:
            borne += poids_dispo * self.objets_tries[j]['ratio']

        return borne


    
    def explorer(self, index, poids_restant, valeur_actuelle, selection_actuelle):
        
        if index == self.nb_objets:
            if valeur_actuelle > self.meilleure_valeur:
                self.meilleure_valeur = valeur_actuelle
            return

        objet = self.objets_tries[index]


        if objet['poids'] <= poids_restant:
            selection_actuelle[index] = 1
            self.explorer(index + 1, poids_restant - objet['poids'], valeur_actuelle + objet['valeur'], selection_actuelle)
            
        selection_actuelle[index] = 0


    def resoudre(self, poids_max):
        
        self.meilleure_valeur = 0
        self.meilleure_selection = [0] * self.nb_objets
        selection_vide = [0] * self.nb_objets
        
        self.explorer(0, poids_max, 0, selection_vide)
        
        return self.meilleure_selection, self.meilleure_valeur
        