#import pandas as pd
import numpy as np
from logic import *
from scipy.spatial import distance

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

###### GESTION DES VARIABLES #######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class VariableUnicity:
    def __init__(self):
        self._variables = {}

    def get(self, name):
        if name not in self._variables:
            self._variables[name] = Variable(name)
        return self._variables[name]

    def get_many(self, *names):
        return [self.get(name) for name in names]

    def __str__(self):
        variables_str = ', '.join(self._variables.keys())
        return f"Variables enregistrées: {variables_str}"
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def list_to_vars(Var_dict,Str_List):                # Cette fonction permet à partir d'une liste de noms de variables (potentiellement contenant '~' comme marqueur de la négation en 1er caractère) de créer une liste de propositions
    temp = np.array(Str_List)
    negations_index = []
    for indice,t in enumerate(temp):                # On regarde toutes les variables qui ont des négations (il peut y avoir plusieurs égations ex: ~~~a)
        i = 0
        negation = False
        while i<len(t) and t[i]=='~':
            i+=1
            negation = not negation
        temp[indice] = temp[indice][i:]             # On enlève les '~' des chaînes de caractères
        if negation:
            negations_index.append(indice)                  # On note l'indice de toutes celles qui correspondent à des négations (par ex, si on a ~~a on ne compte pas a puisque les 2 negations s'annulent)

    P = np.array(Var_dict.get_many(*temp))             # Ici on check si il existe déjà des variables (négation enlevée) avec les noms donnés et on renvoie les variables correspondantes (on ajoute les variables qui n'existaient pas)
    P[negations_index] = [~s for s in P[negations_index]]               # On rajoute la négation sur les variables concernées
    
    return P.tolist()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''               # Version plus générale de negation_equivalence, au lieu de remplacer les négations on inclut toutes les prémices impliquées
def ensemble_premices_equi (premises, W):             # En recevant un vecteur de premises, renvoie toutes les prémises "impliquées" par W
    extended = list(premises.copy())
    changed = True

    while changed:              # On boucle tant que des premises sont ajoutées
        changed = False
        for f in W:
            if isinstance(f, Iff):
                left, right = f.children

                if left in extended and right not in extended:
                    extended.append(right)
                    changed = True
                elif isinstance(left, Not) and left.children[0] in extended and Not(right) not in extended:
                    extended.append(Not(right))
                    changed = True
                elif right in extended and left not in extended:
                    extended.append(left)
                    changed = True
                elif isinstance(right, Not) and right.children[0] in extended and Not(left) not in extended:
                    extended.append(Not(left))
                    changed = True

            elif isinstance(f,Implies):
                left,right = f.children
                if left in extended and right not in extended:
                    extended.append(right)
                    changed = True

            elif isinstance(f,ImpliedBy):
                left,right = f.children
                if right in extended and left not in extended:
                    extended.append(left)
                    changed = True

    return extended

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''               # Pareil mais juste avec les implications (suppose qu'on a traité les synonymes et antonymes avant)
def ensemble_premices_equi2 (premises, W):             # En recevant un vecteur de premises, renvoie toutes les prémises "impliquées" par W
    extended = list(premises.copy())
    changed = True

    while changed:              # On boucle tant que des premises sont ajoutées
        changed = False
        for f in W:
            if isinstance(f,Implies):
                left,right = f.children
                if left in extended and right not in extended:
                    extended.append(right)
                    changed = True

            elif isinstance(f,ImpliedBy):
                left,right = f.children
                if right in extended and left not in extended:
                    extended.append(left)
                    changed = True

    return extended

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#






###### RULE #######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
class Rule:
    def __init__(self,premises, conclusion):
        # A initialiser uniquement dans une Rule_Base

        if not all(isinstance(p, Proposition) for p in premises):
            raise TypeError("premises doit être une liste de propositions")
        
        if not all(isinstance(c, Proposition) for c in conclusion):
            raise TypeError("conclusion doit être une liste de propositions")
        
        self.premises = premises   # Une liste de propositions
        self.conclusion = conclusion   # Une liste de propositions

    def __str__(self):
        premises_str = ' ^ '.join(str(p) for p in self.premises)
        conclusion_str = ' ^ '.join(str(c) for c in self.conclusion)
        return f"Rule: {premises_str} => {conclusion_str}"

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK''' 
def is_a_in_b(short, long,W):
    return all(any(x.is_equivalent(y) for y in ensemble_premices_equi2(long,W)) for x in ensemble_premices_equi2(short,W))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def is_element_in_list(element, liste,W):
    return np.where([x.is_equivalent(element) for x in ensemble_premices_equi2(liste,W)])[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def children_extraction(formula):
    childrens = []
    temp = formula.children
    for i in temp:
        if i.children == []:
            childrens.append(i.name)
        else:
            childrens = childrens+children_extraction(i)
    return childrens

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def update_dict_local (Rb,f):                   # Attention verifier que c'est une équivalence avant
    W = Rb.W
    dict_local = Rb.dict_local
    temp = dict_local
    left,right = f.children
    if len(dict_local) == 0:
        temp.append([left,right])
        return temp
    l = -1
    r = -1
    synonyme = True
    for i,D in enumerate(dict_local):
        if l==-1:
            if is_a_in_b([left], D,W):                # si la partie gauche est dans le vecteur
                if is_a_in_b([right],D,W):                    # et la partie droite
                    return temp             # pas de modifications
                else:
                    synonyme = not synonyme
                    l = i
            elif is_a_in_b([Not(left)], D,W):               # De même avec la négation
                if is_a_in_b([Not(right)],D,W):
                    return temp
                else:
                    l = i
        if r==-1:
            if is_a_in_b([right], D,W):
                if is_a_in_b([left],D,W):
                    return temp
                else:
                    synonyme = not synonyme
                    r = i
            elif is_a_in_b([Not(right)], D,W):               # De même avec la négation
                if is_a_in_b([Not(left)],D,W):
                    return temp
                else:
                    r = i

    if l==-1 and r==-1:
        temp.append([left,right])
        return temp
    if l==-1:
        if synonyme:                    # concrètement si on a détecté 1 seul des 2 bouts, synonyme à la valeur inverse de ..;;
            temp[r] += [Not(left)]
        else:
            temp[r] += [left]
        return temp
    if r==-1:
        if synonyme:
            temp[l] += [Not(right)]
        else:
            temp[l] += [right]
        return temp
    else:
        if synonyme:
            temp[l] += temp[r]              # on joint les vecteurs
            del temp[r]             # On supprime le second
        else:
            for i in temp[r]:
                temp[l] += [Not(i)]             ## on joint les vecteurs
            del temp[r]            # On supprime le second
        return temp
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def synonymes_elimination (premises,dict_local,W):                # On traduit les synonymes
    temp = list(premises.copy())
    for D in dict_local:
        syn = False
        ant = False
        for d in D:
            indices_syn = is_element_in_list(d, temp,W)
            if len(indices_syn)>0 and syn:
                for i in indices_syn:
                    del temp[i]
            if len(indices_syn)>0 and not syn:
                syn = True
                temp[indices_syn[0]] = D[0]
                for i in indices_syn[1:]:
                    del temp[i]
            indices_ant = is_element_in_list(Not(d), temp,W)
            if len(indices_ant)>0 and ant:
                for i in indices_ant:
                    del temp[i]
            if len(indices_ant)>0 and not ant:
                ant = True
                temp[indices_ant[0]] = Not(D[0])
                for i in indices_ant[1:]:
                    del temp[i]
    return temp

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

###### RULEBASE #######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK''' 
class Rule_Base:
    def __init__(self):
        self.premises = []          # Liste unique des prémisses utilisés
        self.conclusions = []           # Liste de toutes les conclusions des règles
        self.rules = []         # toutes les rules
        self.P = []         # Matrice binaire des prémisses de chaque rule
        self.C = []         # Vecteur des conclusions (Propositions)
        self.compteur = 0
        self.Var_dictionnary = VariableUnicity()                # On crée un dictionnaire de toutes les variables utilisés pour s'assurer de leur unicité
        self.W = []
        self.S = []
        self.S_original = []                # Version originale de S (sans les ajoutes des synonyme, négations, etc...) sous forme de string
        self.rules_original = []                # Version originale des règles (sans les ajoutes des synonyme, négations, etc...) sous forme de string
        self.dict_local = []                # liste qui contient des vecteurs de variables équivalentes

    def __str__(self):
        return "\n".join(str(rule) for rule in self.rules)
    
    def all_dictionnary(self):
        return self.Var_dictionnary
    
    def add_W(self,f_string_list):             # ATTENTION, IL FAUT INITIALISER W EN 1ER ET NE PLUS AJOUTER DE REGLES DEDANS                # TEST OK
        for i in f_string_list:
            f = str_to_formula(i,self)
            self.W.append(f)
            if isinstance(f,Iff):
                self.dict_local = update_dict_local (self,f)

    def init_S(self,list_S):                # TEST OK
        self.S_original = ' ^ '.join(str(s) for s in list_S)
        S_no_syn = synonymes_elimination(list_to_vars(self.Var_dictionnary,list_S),self.dict_local,self.W)
        self.S = ensemble_premices_equi2 (S_no_syn,self.W)
        count = 0
        for s in self.S:
            if (not isinstance(s,Not)) and (s not in self.premises):
                self.premises.append(s)
                count += 1
            elif (isinstance(s,Not)) and (s.children[0] not in self.premises):
                self.premises.append(s.children[0])
                count +=1
        
        for vecteur in self.P:              # Mise à jour de la matrice P
            vecteur.extend([0] * count)

    def add_rules(self, list_P, list_C):                # TEST OK
        if self.W == []:
            raise ValueError("Il faut définir W avant d'ajouter des règles")
        if len(list_P) != len(list_C):
            raise ValueError("Nombre de listes de prémises et de conclusions incohérent.")
        for i in range(len(list_P)):
            str1 = ' ^ '.join(str(s) for s in list_P[i])                # On enregistre un string de la règle pour l'affichage (avant l'ajout des synonymes etc.. pour plus de lisibilité)
            str2 = ' ^ '.join(str(s) for s in list_C[i])
            self.rules_original.append(f"{str1} => {str2}")
            
            P = synonymes_elimination (list_to_vars(self.Var_dictionnary,list_P[i]),self.dict_local,self.W)              # On élimine les syonymes/antonymes
            C = synonymes_elimination (list_to_vars(self.Var_dictionnary,list_C[i]),self.dict_local,self.W)

            P_bis = ensemble_premices_equi2(P,self.W)               # on gère les implications
            C_bis = ensemble_premices_equi2(C,self.W)

            for c in C_bis:
                if (not isinstance(c,Not)) and (c not in self.conclusions):               # Mise à jour de self.conclusions
                    self.conclusions.append(c)
                elif (isinstance(c,Not)) and (c.children[0] not in self.conclusions):
                    self.conclusions.append(c.children[0])
                
            count = 0               # compteur du nombre de premises ajoutées
            for p in P_bis:             # Mise à jour de self.premises
                if (not isinstance(p,Not)) and (p not in self.premises):
                    self.premises.append(p)
                    count += 1
                elif (isinstance(p,Not)) and (p.children[0] not in self.premises):
                    self.premises.append(p.children[0])
                    count +=1

            bin_vector = [1 if prem in P_bis else -1 if Not(prem) in P_bis else 0 for prem in self.premises]              # Création du vecteur de la nouvelle règle (1 pour la présence d'un premise, -1 pour la négation d'un premise, 0 sinon)

            for vecteur in self.P:              # Mise à jour de la matrice P
                vecteur.extend([0] * count)

            rule = Rule(P, C)               # Mise à jour des variables de la rule base
            self.rules.append(rule)
            self.P.append(bin_vector)
            self.C.append(C)
            self.compteur += 1

    #def remove_rules(self,l):              # Pas à jour et inutilisé
    #    self.rules = [v for i, v in enumerate(self.rules) if i not in l]
    #    self.C  = [v for i, v in enumerate(self.C) if i not in l]
    #    self.P  = [v for i, v in enumerate(self.P) if i not in l]
    #    self.compteur = self.compteur-1
    
    def inclusion(self, indices):                # TEST OK
        if len(indices) == 0:           # Convention: si le vecteur est vide c'est qu'on veut comparer avec toute les règles
            return [i for i in range(self.compteur) if is_a_in_b(self.rules[i].premises, self.S,self.W)]
        else:
            return [i for i in indices if is_a_in_b(self.rules[i].premises, self.S,self.W)]
        
    def compatibility_matrix(self,indices):                # TEST OK
        n = len(indices)             # indices est un vecteur des indices de toutes les règles dont on veut comparer la compatibilité
        if n>self.compteur:
            raise ValueError("Vous avez appelé plus de règles qu'il n'en existe dans la base")
        compatibility_matrix = np.zeros((n,n))

        for a in range(n):
            for b in range(a+1, n):
                i = indices[a]
                j = indices[b]

                r1 = self.rules[i]
                r2 = self.rules[j]

                if is_a_in_b(r1.premises, r2.premises,self.W):              # On teste si il y a inclusion des premises d'une règle dans l'autre
                    if not self.compatible([r1,r2],conclusions_only=True):              # Est ce que les conclusions sont compatibles?
                        compatibility_matrix[a, b] = 1
                elif is_a_in_b(r2.premises, r1.premises,self.W):               # Si c'est inclus dans l'autre sens on remplit le bas de la matrice,
                    if not self.compatible([r1,r2],conclusions_only=True):              # Est ce que les conclusions sont compatibles?
                        compatibility_matrix[b, a] = 1
        return compatibility_matrix
    
    def dist_hamming(self, indice):                # TEST OK
        P1 = np.atleast_2d(self.P[indice])[0]             # indice est l'indice de la règle qu'on va comparer aux autres
        C1 = self.C[indice]

        C = self.C
        P = np.array(self.P)
        same_concl = []
        for c in C:
            if is_a_in_b(C1, c,self.W) and is_a_in_b(c, C1,self.W):                # Si l'un est inclus dans l'autre et qu'ils ont la même longueur
                same_concl.append(True)
            else:
                same_concl.append(False)

        n = len(self.Var_dictionnary._variables)
        dists = np.full(len(C), n + 1)             # On fixe la distance par défault à delta, si les conclusions des 2 régles sont les mêmes on modifira cette distance

        if np.any(same_concl):
            same_ccl = [i for i, x in enumerate(same_concl) if x == True]
            for i in same_ccl:
                dists[i] = distance.hamming(P1, P[i])*len(P1)               # Règles avec mm conclusion, calcul de distance de Hamming     
        return list(dists)

    #def is_identical(self):                 # PAS A JOUR ET INUTILISE
    #    bin_vector = [1 if prem in self.S else -1 if Not(prem) in self.S else 0 for prem in self.premises]
    #    try:
    #        return self.P.index(bin_vector)
    #    except ValueError:
    #        return -1
        
    def compatible(self,rules,conclusions_only=False,premices_list=None):
        Truth_dict = dictionnaire_eval_rules (self,rules,conclusions_only,premices_list)
        if Truth_dict == -1:
            return False
        W_temp = (self.W).copy()
        for w in self.W:
            if isinstance(w,Iff):
                W_temp.remove(w)                # On a géré au préalable les équivalences, plus besoin de le faire ici
        return all(w.evaluate(**Truth_dict) for w in W_temp)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def dictionnaire_eval_rules (Rb,rules,conclusions_only = False,premices_list=None):                # Quand on veut sélectionner les règles applicables on s'intéresse aux prémices et aux conclusions
    # quand on veut vérifier qu'une règle est l'exception d'une autre on s'intéresse uniquement à la compatibilité des CONCLUSIONS
    Truth_Dict = {p : False for p in Rb.Var_dictionnary._variables}              # On crée un dictionnaire avec toutes les propositions utilisées dans Rb (ne contient pas de négation par construction)
    if conclusions_only == True:                # Cas pour la gestion des exceptions
        Propositions = []
    else:
        Propositions = Rb.S
    if premices_list != None:              # Cas où on a juste une liste de prémices (c'est pas propre à changer plus tard) --> quand on a pas encore ajouté les règles à la base
        Propositions = Propositions + premices_list
    for r in rules:# Liste des propositions utilisées dans S plus les conclusions des 2 règles sélectionnées
        Propositions =  Propositions + r.conclusion
    all_p = ensemble_premices_equi2(Propositions, Rb.W)                # Toutes les propositions engendrées par les propositions de départ
    for s in all_p:
        for s_bis in all_p:
            if s_bis.is_equivalent(Not(s)):             # On teste si la négation de chacunes des propositions est dans le vecteur, si c'est le cas on sort immédiatement de la fonction
                return -1
        if not isinstance(s,Not):
            Truth_Dict[s.name] = True               # Si jamais la proposition n'est pas une négation, on fixe sa valeur à True (cf. HYPOTHESE)
    return Truth_Dict

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



###### SELECTION REGLES ADAPTEES ######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def Select_Rule_web (rulebase,regles_possibles):
        C_matrix = rulebase.compatibility_matrix(regles_possibles)
        rows_to_remove = set(np.where(C_matrix == 1)[0])
        regles_possibles = [r for i, r in enumerate(regles_possibles) if i not in rows_to_remove]               # On va supprimer toutes les règles moins prioritaires
        return regles_possibles

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def select_fct_treshold (Dist_vector,threshold):
    print("Dist_vector",Dist_vector)
    i = Dist_vector.index(0)                # On enlève la règle qu'on compare
    Dist_vector[i] = int(threshold)+1

    D = np.array(Dist_vector)
    return np.where(D < int(threshold))[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def select_fct_minimal (Dist_vector):
    i = Dist_vector.index(0)                # On enlève la règle qu'on compare
    Dist_vector[i] = max(Dist_vector)+1

    D = np.array(Dist_vector)
    return np.where(D == np.min(D))[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def select_fct_treshold_minimal (Dist_vector,threshold):
    i = Dist_vector.index(0)                # On enlève la règle qu'on compare
    Dist_vector[i] = int(threshold)+1

    D = np.array(Dist_vector)
    if np.min(D)<int(threshold):
        retour = np.where(D == np.min(D))[0]
    else:
        retour = []
    return retour

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def scenario_check_web4_test(S, rulebase,deja_appliquees):
    rulebase.init_S(S)
    output = []

    regles_possibles = rulebase.inclusion([])

    temp = regles_possibles.copy()
    for i in regles_possibles:             #Eliminer règles donc la conclusion est incompatible avec la situation (et celles déjà appliquées)
        r  = rulebase.rules[i]
        if is_a_in_b(r.conclusion, rulebase.S,rulebase.W) or (not rulebase.compatible([r])) or (i in deja_appliquees):               #eliminer aussi les règles dont les conclusions sont déjà dans S
            temp.remove(i)
    regles_possibles = temp

    if len(regles_possibles) > 1:
        output.append("\n")
        output.append("Plusieurs règles correspondent à la situation:")
        for i in regles_possibles:
            output.append(f"- Règle {i} : {rulebase.rules_original[i]}")

        C_matrix = rulebase.compatibility_matrix(regles_possibles)
        rows_to_remove = set(np.where(C_matrix == 1)[0])                # suppression des règles moins prioritaires

        for i in rows_to_remove:
            for j in range(len(regles_possibles)):
                if C_matrix[i, j] == 1:
                    r1_index = regles_possibles[i]
                    r2_index = regles_possibles[j]
                    output.append(f"La règle {r2_index} est prioritaire sur la règle {r1_index}, on écarte la règle {r1_index}")
    
        regles_possibles = [r for i, r in enumerate(regles_possibles) if i not in rows_to_remove]
    elif len(regles_possibles) == 1:
        output.append("\n")
        output.append("Voici l'unique règle qui correspond à la situation:")
        output.append(f"- Règle {regles_possibles[0]} : {rulebase.rules_original[regles_possibles[0]]}")

    return {
        "output":output,
        "options": [rulebase.rules_original[i] for i in regles_possibles],
        "indices": regles_possibles
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#def generation_extension ():


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def choix_exception(distance_method, rulebase, selection_fct_and_args,regle_choisie):
    selection_fct = selection_fct_and_args[0]
    args = selection_fct_and_args[1:]

    selected_indices = globals()[selection_fct](getattr(rulebase, distance_method)(regle_choisie), *args)               # on sélectionne les règles dont la distance à regle_choisie satisfait les critères de la fonction de sélection choisie
    indices_similaires,exceptions_associees,adaptations_associees = exceptions(rulebase, selected_indices,rulebase.rules[regle_choisie])                # On filtre et élimine les règles qui n'ont pas d'exceptions
    return {"indices":indices_similaires,
            "options":[rulebase.rules_original[i] for i in indices_similaires],
            "exceptions associées":[[rulebase.rules_original[i] for i in liste] for liste in exceptions_associees],
            "regles_adaptees":adaptations_associees}

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def difference_premises (longue,courte,W):                # Difference entre 2 listes de prémices, avec longue une liste qui contient courte
    diff =[]
    for p in longue:
        if len(is_element_in_list(p, courte,W))==0:
            diff.append(p)
    return diff

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def exceptions(Rb, selected_indices,regle_choisie):
    filtre = []
    excep = []
    adaptations_associees = []
    for i in selected_indices:
        r1 = Rb.rules[i]
        liste_exceptions = []                # liste des exceptions de la règle i
        liste_adaptations = []                # liste des adaptations de la règle i
        for j,r2 in enumerate(Rb.rules):                # On compare avec toutes les autres règles pour détecter les exceptions associées
            if j == i:
                continue
            if is_a_in_b(r1.premises, r2.premises,Rb.W):             # Si on a des prémices incluses dans r2 on étudie la compatibilité des ccl
                if not Rb.compatible([r1,r2],conclusions_only=True):              # Si elles sont incompatibles on sélectionne la règle
                    # On définit ici les caractéristques de l'exception associée, p et c
                    p_adaptation = regle_choisie.premises+difference_premises(r2.premises,r1.premises,Rb.W)
                    c_adaptation = r2.conclusion
                    if is_a_in_b(p_adaptation, Rb.S,Rb.W) and Rb.compatible([],conclusions_only=False,premices_list=c_adaptation):           
                        # Si jamais l'adpatation proposée n'est pas applicable dans la situation, on passe à la suite (prémices pas dans S ou conclusion incompatible avec S)
                        if i not in filtre:
                            filtre.append(i)
                        liste_exceptions.append(j)
                        liste_adaptations.append([p_adaptation,c_adaptation])
        if len(liste_exceptions)>0:
            excep.append(liste_exceptions)
            adaptations_associees.append(liste_adaptations)
    return filtre,excep,adaptations_associees             # On renvoie les règles et leurs exceptions associées

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def init_rule_base2():
    Rb = Rule_Base()
    Rb.add_W([
        "interdit <=> ~ autorisé",
        "~ ( moins_30 & entre30_50 ) & ~ ( moins_30 & plus_50 ) & ~ ( entre30_50 & plus_50 )",
        "gyrophare_allumé >> etat_urgence",
        "etat_urgence >> alarme",
        "gyrophare_allumé << alarme",
        "interdit <=> prohibé"
    ])
    Rb.add_rules(
        [["cycliste", "traverse_parc","écouteurs"],
         ["cycliste", "traverse_parc", "écouteurs","~passants"],
         ["véhicule", "traverse_feu_rouge"],
         ["véhicule", "etat_urgence", "traverse_feu_rouge"],
         ["véhicule", "traverse_parc"],
         ["véhicule", "etat_urgence", "autorisé"]],
        [["interdit"], ["autorisé"],["interdit"], ["autorisé"], ["interdit"],["~amende"]]
    )
    return Rb

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

###### PARSAGE ######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def get_var_from_index(token_to_var, i):                # renvoie la variable qui correspond à l'indice demandé dans le dictionnaire
    for index, var in token_to_var:
        if index == i:
            return var
    return None

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def formula_creator(tokens, token_to_var):                  # va créer une formule à partir d'une liste de tokens et d'une liste de variables (avec leur position)
    stack = []              # Endroit où on va stocker la formule
    i = 0                   # indice d'avancement
    neg = False                 # négation

    while i < len(tokens):                  # On boucle sur les tokens
        token = tokens[i]                   # token actuel

        if token == ")":                    # erreur lors de la fermeture d'une parenthèse sans ouverture préalable
            raise ValueError("Invalid formula: ')' without matching '('")

        elif token == "(":                  # Si on ouvre une parenthèse on va faire un appel récursif sur la partie entre parenthèses
            sub_tokens, j = extract_subtokens(tokens, i)                    # appel de extract_subtokens pour trouver la fin de la parenthèse
            token_to_var_local = token_to_var.copy()                # copie pour ne pas modifier la version générale
            token_to_var_local[:, 0] = token_to_var_local[:, 0] - (i + 1)               # décalage des indices pour correspondre à la partie entre parenthèses
            sub_formula = formula_creator(sub_tokens, token_to_var_local)                  # appel récursif sur la partie entre parenthèses
            stack.append(~sub_formula if neg else sub_formula)                  # On rajoute dans stack la partie de formule qu'on vient de calculer, si il restait une négation identifiée plus tôt on la prends en compte
            neg = False             # ré-initialisation de neg
            i = j               # On saute à l'indice de fin de parenthèse

        elif token == "~":
            neg = not neg               # Si on détecte une négation, on inverse la valeur de neg (ça permet de traiter les cas du type ~~~ qui est équivalent à juste ~)

        elif token in ["&", "|"]:                   # Si on a un opérateur & ou | qui nécessite de connaître la formule à droite et à gauche on calcule la partie droite (on a déjà la gauche dans stack)
            if len(stack) < 1:
                raise ValueError(f"Operator {token} missing left operand")
            left = stack.pop()
            i += 1
            while i < len(tokens) and tokens[i] == "~":                 # cas des négations
                neg = not neg
                i += 1
            if i >= len(tokens):                    # formule incomplète
                raise ValueError(f"Operator {token} missing right operand")
            right, i = eval_subformula(tokens, token_to_var, i, neg)                # on évalue la sous-formule droite
            neg = False
            stack.append(left & right if token == "&" else left | right)                # On ajoute dans stack

        else:
            var = get_var_from_index(token_to_var, i)               # Initialisation de stack avec une variable si la chaîne commence comme ça
            if var is None:
                raise ValueError(f"Unknown variable: '{token}' at index {i}")
            stack.append(~var if neg else var)
            neg = False

        i += 1
    if neg:
        raise ValueError("'~' sans variable")
    if len(stack) != 1:
        raise ValueError("Invalid formula: unable to reduce to single expression")
    return stack[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def extract_subtokens(tokens, start):               # On isole les parties entre parenthèses
    depth = 1
    j = start + 1
    while j < len(tokens):              # depth permet de vérifier qu'on ouvre/ferme le bon nombre de parenthèses
        if tokens[j] == "(":
            depth += 1
        elif tokens[j] == ")":
            depth -= 1
            if depth == 0:              # On sort si on a fermé la parenthèse de départ
                break
        j += 1
    if j == len(tokens):
        raise ValueError("Parenthesis not closed")
    return tokens[start + 1:j], j               # On renvoie la partie entre parenthèses et l'indice de fin de la parenthèse

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def eval_subformula(tokens, token_to_var, i, neg):
    if tokens[i] == "(":                    # Si c'est une parenthèse on fait juste comme dans le cas général des parenthèses pour calculer tout le bloc
        sub_tokens, j = extract_subtokens(tokens, i)
        token_to_var_local = token_to_var.copy()
        token_to_var_local[:, 0] = token_to_var_local[:, 0] - (i + 1)
        sub_formula = formula_creator(sub_tokens, token_to_var_local)
        return ~sub_formula if neg else sub_formula, j
    else:                   # On ne gère pas le cas des négations vu qu'il a été traité avant l'appel de la fonction
        var = get_var_from_index(token_to_var, i)
        if var is None:
            raise ValueError(f"Variable at position {i} not found")
        return ~var if neg else var, i                  # On renvoie la sous-formule et l'indice de fin
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def str_to_formula(formula_str, Rb):                # à partir d'un string crée une formule pour W
    tokens = formula_str.split(" ")             # le séparateur du string est le caractère espace

    if sum(tokens.count(op) for op in ["<=>", ">>", "<<"]) > 1:                 # On considère des formules avec 1 seul opérateur du type <=> >> << maximum
        raise ValueError("Only one main binary operator ('<=>', '>>', '<<') allowed")

    var_tokens = []             # On va créer des sous listes avec les variables qu'on va utiliser et leur position dans la chaîne de caractères
    var_indices = []

    for i, token in enumerate(tokens):
        if token not in ["<=>", "~", ">>", "<<", "&", "|", "(", ")"]:                   # Si le token n'est pas un opérateur, on crèe une variable
            var_tokens.append(token)
            var_indices.append(i)

    var_objs = list_to_vars(Rb.Var_dictionnary, var_tokens)                 # On crée les variables correspondantes

    token_to_var = np.array(list(zip(var_indices, var_objs)), dtype=object)                 # np.array qui contient des tuples associant la variable à sa position dans la chaîne de caractères

    if "<=>" in tokens:                 # On regarde les 3 cas possibles d'opérateurs principaux, on divise à droite et à gauche de l'opérateur puis on appelle formula_creator poru créer les formules correspondantes
        idx = tokens.index("<=>")
        left = formula_creator(tokens[:idx], token_to_var)
        token_to_var[:, 0] = token_to_var[:, 0] - (idx + 1)  # Ajuste les indices pour le sous-tableau de droite
        right = formula_creator(tokens[idx+1:], token_to_var)
        return left.iff(right)

    elif ">>" in tokens:
        idx = tokens.index(">>")
        left = formula_creator(tokens[:idx], token_to_var)
        token_to_var[:, 0] = token_to_var[:, 0] - (idx + 1)
        right = formula_creator(tokens[idx+1:], token_to_var)
        return left >> right

    elif "<<" in tokens:
        idx = tokens.index("<<")
        left = formula_creator(tokens[:idx], token_to_var)
        token_to_var[:, 0] = token_to_var[:, 0] - (idx + 1)
        right = formula_creator(tokens[idx+1:], token_to_var)
        return left << right

    else:
        return formula_creator(tokens, token_to_var)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#