import numpy as np
from logic import *
from scipy.spatial import distance
import shutil
from flask_babel import gettext as _  
import os
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

###### GESTION DES VARIABLES #######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class VariableUnicity:              # Variable dictionary, ensures unicity (two different variables cannot have the same name)
    def __init__(self):
        self._variables = {}

    def get(self, name):                # From a name, returns the associated variable (creates it if necessary)
        if name not in self._variables:
            self._variables[name] = Variable(name)
        return self._variables[name]

    def get_many(self, *names):             # Same from a list of names
        return [self.get(name) for name in names]

    def __str__(self):
        variables_str = ', '.join(self._variables.keys())
        return f"Variables enregistrées: {variables_str}"
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
# From a list of variables names creates a list of propositions (potentially containing ‘~’ as a negation marker in the first character)
def list_to_vars(Var_dict,Str_List):                
    temp = np.array(Str_List)
    negations_index = []
    for indice,t in enumerate(temp):                # We look at all variables that have negations (there may be several negations, e.g., ~~~a)
        i = 0
        negation = False
        while i<len(t) and t[i]=='~':
            i+=1
            negation = not negation                 # Loop to know if the variable is a negation (~~a -> a)
        temp[indice] = temp[indice][i:]             # Remove the ‘~’ characters from the strings
        if negation:
            negations_index.append(indice)                  # We note the index of all those that correspond to negations (e.g., if we have ~~a, we do not count 'a' since the two negations cancel each other out)

    P = np.array(Var_dict.get_many(*temp))             # Here we check if variables already exist (negation removed) with the given names and return the corresponding variables (we add the variables that did not exist)
    P[negations_index] = [~s for s in P[negations_index]]               # We add the negation to the relevant variables
    
    return P.tolist()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def ensemble_premisses_equi2 (premisses, W):             # When given a vector of premises, returns all premises “implied” by W (synonyms and antonyms are treated before)
    extended = list(premisses.copy())
    changed = True

    while changed:              # We loop as long as premises are added
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
    def __init__(self,premisses, conclusion):
        # To be initialized only in a Rule_Base

        if not all(isinstance(p, Proposition) for p in premisses):
            raise TypeError("premisses doit être une liste de propositions")
        
        if not all(isinstance(c, Proposition) for c in conclusion):
            raise TypeError("conclusion doit être une liste de propositions")
        
        self.premisses = premisses   # propositions list
        self.conclusion = conclusion   # propositions list

    def __str__(self):
        premisses_str = ' ^ '.join(str(p) for p in self.premisses)
        conclusion_str = ' ^ '.join(str(c) for c in self.conclusion)
        return f"{premisses_str} => {conclusion_str}"

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK''' 
def is_a_in_b(short, long,W):               # return True if the first list of premises is a sub-list of the second list, False else
    return all(any(x.is_equivalent(y) for y in ensemble_premisses_equi2(long,W)) for x in ensemble_premisses_equi2(short,W))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def is_element_in_list(element, liste,W):               # return the index of the element in the list if it is an element
    return np.where([x.is_equivalent(element) for x in ensemble_premisses_equi2(liste,W)])[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def children_extraction(formula):               # from a formula extracts all premises used (without negations or anything)
    if not formula.children:
        return [formula.name]
    return [name for child in formula.children for name in children_extraction(child)]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def update_dict_local (Rb,f):                   # Given an equivalence formula, updates the synonyms/antonyms dictionnary
    """
    Update the equivalence classes of synonyms/antonyms in Rb.dict_local
    based on a new equivalence formula `f`.

    Each equivalence class is a list of expressions that are considered
    equivalent (or negated equivalents, depending on context).

    Returns:
        The updated dict_local list with merged or extended equivalence classes.

    Notes:
        - If both terms already belong to the same class, nothing changes.
        - If neither term is found, a new class is created.
        - If only one side is found, the other is added (possibly as a negation).
        - If both are found in different classes, those classes are merged.
        - The `synonym_flag` helps decide whether to insert negations when merging.
    """

    W = Rb.W
    classes = Rb.dict_local
    left,right = f.children             # both sides of the equivalence

    if not classes:
        # Initialize first equivalence class
        classes.append([left,right])
        return classes
    
    l_class, r_class = -1, -1                   # indices of classes for left/right

    synonym_flag = True             # determines whether to negate terms when merging

    for i,eq_class in enumerate(classes):
        # Locate the class containing left or its negation
        if l_class==-1:
            if is_a_in_b([left], eq_class,W):
                if is_a_in_b([right],eq_class,W): 
                    return classes             # both already present
                synonym_flag = not synonym_flag
                l_class = i 
            elif is_a_in_b([Not(left)], eq_class,W): 
                if is_a_in_b([Not(right)],eq_class,W):
                    return classes
                l_class = i

        # Locate the class containing right or its negation
        if r_class==-1:
            if is_a_in_b([right], eq_class,W):
                if is_a_in_b([left],eq_class,W):
                    return classes
                else:
                    synonym_flag = not synonym_flag
                    r_class = i
            elif is_a_in_b([Not(right)], eq_class,W):
                if is_a_in_b([Not(left)],eq_class,W):
                    return classes
                else:
                    r_class = i
    # Case 1: Neither side found → new equivalence class
    if l_class==-1 and r_class==-1:
        classes.append([left,right])
        return classes
    
    # Case 2: Only right side found → add left
    if l_class==-1:
        classes[r_class].append(Not(left) if synonym_flag else left)
        return classes
    
    # Case 3: Only left side found → add right
    if r_class==-1:
        classes[l_class].append(Not(right) if synonym_flag else right)
        return classes
    
    # Case 4: Both sides found in different classes → merge
    if synonym_flag:
        classes[l_class].extend(classes[r_class])
    else:
        classes[l_class].extend(Not(term) for term in classes[r_class])
    del classes[r_class]
    return classes
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def synonymes_elimination(premises, dict_local, W):
    """
    Normalize a list of premises by eliminating synonyms and antonyms
    based on equivalence classes.

    Returns:
        list: simplified list of premises where synonyms are merged
              and only one representative per equivalence class remains.
    """
    premises = list(premises)

    for eq_class in dict_local:
        found_syn = False  # True if we've already replaced a synonym (then we will delete all further synonyms instead of traducing them)
        found_ant = False  # True if we've already replaced an antonym (same)

        for element in eq_class:
            # --- Handle synonyms ---
            indices_syn = is_element_in_list(element, premises, W)
            if indices_syn.size > 0:
                if found_syn:
                    # Remove duplicates of the synonym
                    for i in reversed(indices_syn):
                        del premises[i]
                else:
                    # Replace the first occurrence with the class representative
                    premises[indices_syn[0]] = eq_class[0]
                    for i in reversed(indices_syn[1:]):             # delete further occurences
                        del premises[i]
                    found_syn = True

            # --- Handle antonyms ---
            indices_ant = is_element_in_list(Not(element), premises, W)
            if indices_ant.size > 0:
                if found_ant:
                    for i in reversed(indices_ant):
                        del premises[i]
                else:
                    premises[indices_ant[0]] = Not(eq_class[0])
                    for i in reversed(indices_ant[1:]):
                        del premises[i]
                    found_ant = True
    return premises

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

###### RULEBASE #######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK''' 
class Rule_Base:
    def __init__(self):
        self.premisses = []          # List of all premises used
        self.conclusions = []           # List of all conclusions used
        self.rules = []         # All rules
        self.P = []         # Binary matrix of each rules premises
        self.C = []         # Conclusions vector (Propositions) for each rule
        self.compteur = 0
        self.Var_dictionnary = VariableUnicity()                # Variables unicity dictionnary
        self.W = []             # Knowledge Base
        self.S = []             # scenario decomposition
        self.S_original = []                # Initial version of S (string)
        self.rules_original = []                # Initial version of the rules (string) before adding implications, etc...
        self.dict_local = []                # IMPORTANT : list which contains all the information about synonyms and antonyms (<=>)

    def __str__(self):
        return "\n".join(str(rule) for rule in self.rules)
    
    def all_dictionnary(self):
        return self.Var_dictionnary
    
    def add_W(self,f_string_list):             # Knowledge Base Initialisation
                        # CAREFUL, INIT W FIRST, THEN ADD RULES TO THE RB, DO NOT MODIFIY W AFTER ADDING RULES TO THE RB
        for i in f_string_list:
            f = str_to_formula(i,self)              # from a string returns a formula
            self.W.append(f)
            if isinstance(f,Iff):
                self.dict_local = update_dict_local (self,f)                # Updates synonyms/antonyms dict

    def init_S(self,list_S):                # TEST OK
        self.S_original = ' ^ '.join(str(s) for s in list_S)
        S_no_syn = synonymes_elimination(list_to_vars(self.Var_dictionnary,list_S),self.dict_local,self.W)
        self.S = ensemble_premisses_equi2 (S_no_syn,self.W)
        count = 0
        for s in self.S:
            if (not isinstance(s,Not)) and (s not in self.premisses):
                self.premisses.append(s)
                count += 1
            elif (isinstance(s,Not)) and (s.children[0] not in self.premisses):
                self.premisses.append(s.children[0])
                count +=1
        
        for vecteur in self.P:              # Updating P matrix
            vecteur.extend([0] * count)

    def add_rules(self, list_P, list_C):                # TEST OK
        '''
        Add new rules to the rule base.
        
        Each rule is defined by a list of premises (list_P[i]) and a list of conclusions (list_C[i]).
        Synonyms/antonyms are normalized, implications are expanded, and the internal
        structures (premises, conclusions, matrices P and C, counters) are updated.
        '''
        if self.W == []:
            raise ValueError("Il faut définir W avant d'ajouter des règles")
        if len(list_P) != len(list_C):
            raise ValueError("Nombre de listes de prémises et de conclusions incohérent.")
        
        for P_raw, C_raw in zip(list_P, list_C):
            # Human-readable rule string (before synonym/implication processing)
            str1 = ' ^ '.join(str(s) for s in P_raw)
            str2 = ' ^ '.join(str(s) for s in C_raw)
            self.rules_original.append(f"{str1} => {str2}")
            
            # Normalize premises and conclusions (eliminate synonyms/antonyms)
            P = synonymes_elimination (list_to_vars(self.Var_dictionnary,P_raw),self.dict_local,self.W)
            C = synonymes_elimination (list_to_vars(self.Var_dictionnary,C_raw),self.dict_local,self.W)

            # Expand using implications
            P_expanded = ensemble_premisses_equi2(P,self.W)
            C_expanded = ensemble_premisses_equi2(C,self.W)

            # --- Update conclusions ---
            for c in C_expanded:
                base = c if not isinstance(c, Not) else c.children[0]
                if base not in self.conclusions:
                    self.conclusions.append(base)
            
            # --- Update premises ---
            count = 0
            for p in P_expanded:
                base = p if not isinstance(p, Not) else p.children[0]
                if base not in self.premisses:
                    self.premisses.append(base)
                    count +=1

            # --- Build binary vector for the new rule ---
            bin_vector = [1 if prem in P_expanded else -1 if Not(prem) in P_expanded else 0 for prem in self.premisses]

            for vecteur in self.P:                  # Extend existing rows of P for the new premises added
                vecteur.extend([0] * count)

            rule = Rule(P, C)               # --- Final update ---
            self.rules.append(rule)
            self.P.append(bin_vector)
            self.C.append(C)
            self.compteur += 1
    
    def inclusion(self, indices):                   # which rules premises are included in the situation S?
        if len(indices) == 0:           # Convention: if the vector is empty, it means that we want to compare with all the rules.
            return [i for i in range(self.compteur) if is_a_in_b(self.rules[i].premisses, self.S,self.W)]
        else:
            return [i for i in indices if is_a_in_b(self.rules[i].premisses, self.S,self.W)]
        
    def compatibility_matrix(self,indices):
        '''
            Build a compatibility matrix for a subset of rules.

            The matrix entry [i, j] = 1 means:
                - Premises of rule[i] are included in premises of rule[j], AND
                - Their conclusions are not compatible.
        '''
        n = len(indices)             # indices est un vecteur des indices de toutes les règles dont on veut comparer la compatibilité
        if n>self.compteur:
            raise ValueError("Vous avez appelé plus de règles qu'il n'en existe dans la base")
        compatibility_matrix = np.zeros((n,n))

        for a in range(n):
            for b in range(a+1, n):             # Compare all unique pairs (a < b)
                i,j = indices[a], indices[b]
                r1, r2 = self.rules[i], self.rules[j]
                
                if is_a_in_b(r1.premisses, r2.premisses,self.W):                # Check inclusion of premises one way or the other
                    if not self.compatible([r1,r2],conclusions_only=True):
                        compatibility_matrix[a, b] = 1
                elif is_a_in_b(r2.premisses, r1.premisses,self.W):
                    if not self.compatible([r1,r2],conclusions_only=True):
                        compatibility_matrix[b, a] = 1
        return compatibility_matrix
    
    def dist_hamming(self, indice):             # Computes Hamming distance
        P1 = np.atleast_2d(self.P[indice])[0]             # indice -> index of the rule we are gonna compare to the others
        C1 = self.C[indice]

        C = self.C
        P = np.array(self.P)
        same_concl = []
        for c in C:
            if is_a_in_b(C1, c,self.W) and is_a_in_b(c, C1,self.W):                # one is included in the other and they have the same size -> equal
                same_concl.append(True)
            else:
                same_concl.append(False)

        n = len(self.Var_dictionnary._variables)
        dists = np.full(len(C), n + 1)             # delta default distance, if both rules have the same conclusion we calculate Hamming distance

        if np.any(same_concl):
            same_ccl = [i for i, x in enumerate(same_concl) if x == True]
            for i in same_ccl:
                dists[i] = distance.hamming(P1, P[i])*len(P1)  
        return list(dists)
        
    def compatible(self,rules,conclusions_only=False,premisses_list=None):              # compatibility bewteen premises lists or rules
        Truth_dict = dictionnaire_eval_rules (self,rules,conclusions_only,premisses_list)
        if Truth_dict == -1:
            return False
        W_temp = (self.W).copy()
        for w in self.W:
            if isinstance(w,Iff):
                W_temp.remove(w)                # equivalence are already dealt with (eq_classes)
        return all(w.evaluate(**Truth_dict) for w in W_temp)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def dictionnaire_eval_rules (Rb,rules,conclusions_only = False,premisses_list=None): 
    # case 1: check which rules are applicable: we check the compatibility of premises and conclusions
    # case 2:  in the case of verifying that one rule is an exception to another we only look at the comppatibility of CONCLUSIONS
    Truth_Dict = {p : False for p in Rb.Var_dictionnary._variables}              # dict w/ all propositions used in RB (no negations by construction)
    if conclusions_only == True:                # exceptions
        Propositions = []
    else:               # Applicable rules
        Propositions = Rb.S
    if premisses_list != None:              # Case when creating a new exception, the rule is not yet in the base so we add "manually" the new conclusions
        Propositions = Propositions + premisses_list
    for r in rules:             # List of propositions used in S + in rules
        Propositions =  Propositions + r.conclusion
    all_p = ensemble_premisses_equi2(Propositions, Rb.W)                # All implied propositions
    for s in all_p:
        for s_bis in all_p:
            if s_bis.is_equivalent(Not(s)):             # first test: is the negation of the premises in the vector (if yes we quit instantly)
                return -1
        if not isinstance(s,Not):
            Truth_Dict[s.name] = True               # If the proposition is not a negation we set its value to True (cf. HYPOTHESE)
    return Truth_Dict

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#



###### SELECTION REGLES ADAPTEES ######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def select_fct_treshold (Dist_vector,threshold):
    i = Dist_vector.index(0)                # we do not calculate "self" distance because it is useless for applications
    Dist_vector[i] = int(threshold)+1

    D = np.array(Dist_vector)
    return np.where(D < int(threshold))[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def select_fct_minimal (Dist_vector):
    i = Dist_vector.index(0)                # we do not calculate "self" distance because it is useless for applications
    Dist_vector[i] = max(Dist_vector)+1

    D = np.array(Dist_vector)
    return np.where(D == np.min(D))[0]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def select_fct_treshold_minimal (Dist_vector,threshold):
    i = Dist_vector.index(0)                # we do not calculate "self" distance because it is useless for applications
    Dist_vector[i] = int(threshold)+1

    D = np.array(Dist_vector)
    if np.min(D)<int(threshold):
        retour = np.where(D == np.min(D))[0]
    else:
        retour = []
    return retour

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def scenario_check_web4_test(S, rulebase,deja_appliquees,premier_log):
    """
    Evaluate which rules in the rulebase are applicable to a given situation S.

    Returns:
        dict with:
            - "output": list[str], human-readable log of reasoning steps
            - "options": list[str], textual representations of rules that can be applied
            - "indices": list[int], indices of the applicable rules
    """
    rulebase.init_S(S)
    output = []

    regles_possibles = rulebase.inclusion([])               # Start from all rules whose premises are included in the situation

    if premier_log:
        output.append(f" {_('Génération d\'une extension :')} ")

    temp = regles_possibles.copy()
    for i in regles_possibles:             #Elimminate rules which conclusion is incompatible and the already applied ones + the ones whose conclusion is already in S
        r  = rulebase.rules[i]
        if is_a_in_b(r.conclusion, rulebase.S,rulebase.W) or (not rulebase.compatible([r])) or (i in deja_appliquees):
            temp.remove(i)
    regles_possibles = temp

    if len(regles_possibles) >= 1:
        output.append("\n")
        output.append( f"{_("Règles applicables :")}")
        for i in regles_possibles:
            output.append(f"- {_('Règle')} {i} : {rulebase.rules_original[i]}")

        C_matrix = rulebase.compatibility_matrix(regles_possibles)
        rows_to_remove = set(np.where(C_matrix == 1)[0])                # eliminate rules based on priority

        for i in rows_to_remove:
            for j in range(len(regles_possibles)):
                if C_matrix[i, j] == 1:
                    r1_index = regles_possibles[i]
                    r2_index = regles_possibles[j]
                    output.append(f" {_('La règle')} {r2_index} {_('est prioritaire sur la règle')} {r1_index}, {_('on écarte la règle')} {r1_index}")
    
        regles_possibles = [r for i, r in enumerate(regles_possibles) if i not in rows_to_remove]

    return {
        "output":output,
        "options": [rulebase.rules_original[i] for i in regles_possibles],
        "indices": regles_possibles
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def choix_exception(distance_method, rulebase, selection_fct_and_args,regle_choisie):
    """
    Evaluate which exceptions can be created for a selected rule.

    """
    selection_fct = selection_fct_and_args[0]
    args = selection_fct_and_args[1:]

    selected_indices = globals()[selection_fct](getattr(rulebase, distance_method)(regle_choisie), *args)               
    # select rules which for the distnace to regle_choisie satisgies the criterias of the selection function
    indices_similaires,exceptions_associees,adaptations_associees = exceptions(rulebase, selected_indices,rulebase.rules[regle_choisie])                
    # Filter and eliminate rules with no exceptions
    options = [rulebase.rules_original[i] for i in indices_similaires]
    output=""
    if options == []:
        output = f"\n \n {_('Aucune des règles de la base n\'est suffisamment proche pour proposer une adaptation d\'exception')} "
    return {"indices":indices_similaires,
            "options":options,
            "exceptions associées":[[rulebase.rules_original[i] for i in liste] for liste in exceptions_associees],
            "regles_adaptees":adaptations_associees,
            "output":output}

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def difference_premisses (longue,courte,W):                # Difference between 2 premises list (short list is a sublist of long list)
    diff =[]
    for p in longue:
        if len(is_element_in_list(p, courte,W))==0:
            diff.append(p)
    return diff

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def exceptions(Rb, selected_indices, regle_choisie):
    # from a chosen rule, searches for rules with exceptions to adapt to the chosen rule
    filtre = []
    excep = []
    adaptations_associees = []
    for i in selected_indices:
        r1 = Rb.rules[i]
        liste_exceptions = []      # list of exceptions for rule i
        liste_adaptations = []     # list of adaptations for rule i
        for j, r2 in enumerate(Rb.rules):   # compare with all other rules to detect associated exceptions
            if j == i:
                continue
            if is_a_in_b(r1.premisses, r2.premisses, Rb.W):   # if r1’s premises are included in r2’s premises, check compatibility of conclusions
                if not Rb.compatible([r1, r2], conclusions_only=True):   # if conclusions are incompatible, select this rule
                    # define characteristics of the associated exception: premises (p) and conclusion (c)
                    p_adaptation = regle_choisie.premisses + difference_premisses(r2.premisses, r1.premisses, Rb.W)
                    c_adaptation = r2.conclusion
                    if is_a_in_b(p_adaptation, Rb.S, Rb.W) and Rb.compatible([], conclusions_only=False, premisses_list=c_adaptation):
                        # if the proposed adaptation cannot be applied in the situation, skip (premises not in S or conclusion incompatible with S)
                        if i not in filtre:
                            filtre.append(i)
                        liste_exceptions.append(j)
                        liste_adaptations.append([p_adaptation, c_adaptation])
        if len(liste_exceptions) > 0:
            excep.append(liste_exceptions)
            adaptations_associees.append(liste_adaptations)
    return filtre, excep, adaptations_associees   # return the rules and their associated exceptions

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

###### PARSAGE ######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def get_var_from_index(token_to_var, i):                # returns the variable corresponding to the requested index in the dictionary
    for index, var in token_to_var:
        if index == i:
            return var
    return None

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def formula_creator(tokens, token_to_var):                  # creates a formula from a list of tokens and a list of variables (with their positions)
    stack = []              # place where we store the formula
    i = 0                   # current index
    neg = False                 # negation flag

    while i < len(tokens):                  # loop over the tokens
        token = tokens[i]                   # current token

        if token == ")":                    # error: closing parenthesis without a matching opening parenthesis
            raise ValueError("Invalid formula: ')' without matching '('")

        elif token == "(":   # if we open a parenthesis, we make a recursive call on the part inside the parentheses
            sub_tokens, j = extract_subtokens(tokens, i)   # call extract_subtokens to find the matching closing parenthesis
            token_to_var_local = token_to_var.copy()       # copy to avoid modifying the main mapping
            token_to_var_local[:, 0] = token_to_var_local[:, 0] - (i + 1)   # shift indices to match the sub-part inside the parentheses
            sub_formula = formula_creator(sub_tokens, token_to_var_local)   # recursive call on the sub-part
            stack.append(~sub_formula if neg else sub_formula)   # add the computed sub-formula to stack, applying negation if previously set
            neg = False             # reset neg flag
            i = j                   # jump to the index after the closing parenthesis

        elif token == "~":
            neg = not neg               # toggle negation flag (handles cases like ~~~ equivalent to just ~)

        elif token in ["&", "|"]:                    # if we have a binary operator (& or |), compute the right-hand side (left is already in stack)
            if len(stack) < 1:
                raise ValueError(f"Operator {token} missing left operand")
            left = stack.pop()
            i += 1
            while i < len(tokens) and tokens[i] == "~":                 # handle consecutive negations
                neg = not neg
                i += 1
            if i >= len(tokens):                     # incomplete formula
                raise ValueError(f"Operator {token} missing right operand")
            right, i = eval_subformula(tokens, token_to_var, i, neg)                # evaluate right-hand sub-formula
            neg = False
            stack.append(left & right if token == "&" else left | right)                # push result into stack

        else:
            var = get_var_from_index(token_to_var, i)               # initialize stack with a variable if formula starts with it
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
def extract_subtokens(tokens, start):               # Isolates parts between parenthesis
    depth = 1
    j = start + 1
    while j < len(tokens):              # depth -> verify we open/close the right number of parenthesis 
        if tokens[j] == "(":
            depth += 1
        elif tokens[j] == ")":
            depth -= 1
            if depth == 0:              # Break if first parenthesis closed
                break
        j += 1
    if j == len(tokens):
        raise ValueError("Parenthesis not closed")
    return tokens[start + 1:j], j               # Returns part between parenthesis and the index of the end of the parenthesis
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''TEST OK'''
def eval_subformula(tokens, token_to_var, i, neg):
    if tokens[i] == "(":                    # If it is a parenthesis, just do as in the general case of parenthesis to calculate the whole bloc
        sub_tokens, j = extract_subtokens(tokens, i)
        token_to_var_local = token_to_var.copy()
        token_to_var_local[:, 0] = token_to_var_local[:, 0] - (i + 1)
        sub_formula = formula_creator(sub_tokens, token_to_var_local)
        return ~sub_formula if neg else sub_formula, j
    else:                   # The negation case is not treated here because it has been done before calling this func
        var = get_var_from_index(token_to_var, i)
        if var is None:
            raise ValueError(f"Variable at position {i} not found")
        return ~var if neg else var, i                  # Returns the sub-forumla and the end index
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def str_to_formula(formula_str, Rb):                # from a string, create a formula for W
    tokens = formula_str.split(" ")                 # the string separator is the space character

    if sum(tokens.count(op) for op in ["<=>", ">>", "<<"]) > 1:   # We only allow formulas with at most 1 main operator of type <=> >> <<
        raise ValueError("Only one main binary operator ('<=>', '>>', '<<') allowed")

    var_tokens = []             # We'll create sub-lists with the variables to use and their positions in the token list
    var_indices = []

    for i, token in enumerate(tokens):
        if token not in ["<=>", "~", ">>", "<<", "&", "|", "(", ")"]:   # If the token is not an operator, we create a variable
            var_tokens.append(token)
            var_indices.append(i)

    var_objs = list_to_vars(Rb.Var_dictionnary, var_tokens)   # Create the corresponding variable objects

    token_to_var = np.array(list(zip(var_indices, var_objs)), dtype=object)   # np.array containing tuples associating each variable with its position in the token list

    if "<=>" in tokens:     # Check the 3 possible cases for main operators, split left and right of the operator, then call formula_creator to build the corresponding formulas
        idx = tokens.index("<=>")
        left = formula_creator(tokens[:idx], token_to_var)
        token_to_var[:, 0] = token_to_var[:, 0] - (idx + 1)  # Adjust indices for the right sub-array
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

###### Fonctions de app.py ######

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def call_llm(prompt,MODELS,clients,session,first_api,noms):             # From the models available (at the moment Ollama and Mistral if both keys are in the keys file)
    # calls the llm selected with a prompt given as arg, if the llm  is out of capacity, switchs to the other one, etc...
    # Default api (first_api) is set as Ollama by default
    api_order = [session.get("selected_api", first_api)]+noms           
    tried = set()
    for api_name in api_order:
        if api_name in tried:
            continue
        tried.add(api_name)
        try:
            if api_name == "Mistral":
                response = clients[0].chat.complete(
                    model=MODELS[0],
                    messages=[{"role": "user", "content": prompt}])
            else:
                response = clients[1].chat.completions.create(
                    model=MODELS[1],
                    messages=[{"role": "user", "content": prompt}])
            session["selected_api"] = api_name
            return response.choices[0].message.content
        except Exception as e:
            if "capacity" in str(e).lower() or "rate limit" in str(e).lower():
                continue
            raise e
    raise RuntimeError("Aucune API disponible actuellement.")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_files(session):             # returns the path depending on the language selected
    # W_files and RB_files are the list of available files for Knowledge Base and Rule Base. They are extracted from all files in the associated directory
    # So just put the file in the right directory if you want to add one, there are 2 directory, one for each language
    # Be careful if a language were to be added, the name of the directory must be the same as the babel name for the language (en: english, fr: french, etc...)
    lang = session.get("lang", "fr")
    base_rb = lang+"/RB/"
    base_w = lang+"/W/"

    txt_files_rb = [f for f in os.listdir(base_rb) if f.endswith(".txt")]                  # All .txt files from rb
    txt_names_rb = [p[:-4] for p in txt_files_rb]             # Remove all .txt for labels
    paths_rb = [base_rb+p for p in txt_files_rb]
    default_rb = "RB_test"
    default_rb_path = f"{base_rb}{default_rb}.txt"

    txt_files_w = [f for f in os.listdir(base_w) if f.endswith(".txt")]                  # Tout les fichiers .txt d rb
    txt_names_w = [p[:-4] for p in txt_files_w]             # On enlève le .txt pour les labels
    paths_w = [base_w+p for p in txt_files_w]
    default_w = "W_test"
    default_w_path = f"{base_w}{default_w}.txt"
    
    W_files = dict(zip(txt_names_w, paths_w))
    RB_files = dict(zip(txt_names_rb, paths_rb))

    RB_files = {default_rb: default_rb_path, **{k: v for k, v in RB_files.items() if k != default_rb}}
    W_files = {default_w: default_w_path, **{k: v for k, v in W_files.items() if k != default_w}}

    return default_w_path, default_rb_path, W_files, RB_files

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_distance_method():                  # only so that it is not loaded initially and the translation is done correctly
    return {_("Distance de Hamming"): "dist_hamming"}

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_selection_method():                  # only so that it is not loaded initially and the translation is done correctly
    SELECTION_METHODS = {_("Seuil"): "select_fct_treshold",
                         _('Minimale'):"select_fct_minimal",
                         _('Seuil Minimal'):"select_fct_treshold_minimal"}
    return SELECTION_METHODS

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_log(key):               # Dictionnary for the log, the traduction is not working in the app.py file because it is loaded too early, so we call this function
    dic = {1 : _("Aucun scénario fourni."),
           2 : _('Application de la règle'),
           3 : _('Aucune des exceptions proposée n\'a été validée'),
           4 : _('Création d\'une exception'),
           5 : _('à la règle'),
           6 : _('Suite à la création réussie d\'une exception à la règle'),
           7 : _('on annule son application pour poursuivre la génération d\'une extension')}
    return dic[key]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def init_RB(session):               # Initialisation of the Rule Base
    default_w_path, default_rb_path, W_files, RB_files = get_files(session)             # default paths for W and Rb
    Bool = session.get("upload_init", False)                # this value is set at True when the users selectes manually W and the Rb
    Rb = Rule_Base()
    w_path = session.get("selected_w_path", default_w_path)            # path for W
    if not Bool:                # If the user already selected a Rule Base, this step is already done (creating a working RB file, etc...)
        shutil.copy(default_rb_path, "uploads/RB_working.txt")                  # copy of the RB file into a working copy
        session["selected_rb_path"] = "uploads/RB_working.txt"
        session["original_rb_path"] = default_rb_path
        session["upload_init"]=True
        session["selected_w_path"] = w_path

    rb_path = "uploads/RB_working.txt"              # path for RB
    P=[]
    C=[]
    with open(rb_path, "r", encoding="utf-8") as f:             # Reads the lines from the files and init W and RB with the corresponding info
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # Ignore blank lines or comments
                continue
            if ">" in line:
                gauche, droite = line.split(">", 1)  # Split into two parts only
                conditions = gauche.strip().split()
                conclusions = droite.strip().split()

                P.append(conditions)
                C.append(conclusions)
            else:
                print("Ligne ignorée (pas de '>'):", line)
    with open(w_path, "r", encoding="utf-8") as inp:
        W = list(inp.read().splitlines())
    Rb.add_W(W)
    Rb.add_rules(P, C)
    return Rb

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_prompt(scenario,premises,session):              # from a list of premises and the scenario returns the prompt adapted to the language selected
    lang = session.get("lang", "fr")
    if lang == "fr":
        prompt = ("Tu es un expert des textes juridiques et de la décomposition de situations juridiques en prémisses"
        +"Décompose le scénario sous la forme d'une liste de prémisses, en utilisant le caractère ; comme séparateur dans ton retour."
        +"Voici un exemple de scénario, 'Une voiture a traversé un feu rouge', le résultat attendu serait, 'véhicule;traverse_feu_rouge': \n " 
        +"Scénario: \n"+ scenario 
        +"\n Liste des prémisses déjà en mémoire qui peuvent être réutilisés si le scénario comporte des éléments similaires:"
        +premises
        +"\n Ne crée de nouveau prémisse que si c'est nécessaire."
        +"\n Ne rajoute des prémisses que lorsque tu as de l'information explicite, ne fait pas d'inférence. "
        +"\n Par exemple une ambulance n'est pas forcément en état d'urgence et n'a PAS FORCEMENT son gyrophare allumé! C'est le cas uniquement si c'est PRECISE, généralise cet exemple à tout les prémisses"
        +"\n Ton retour ne doit comporter que la liste des prémisses correspondant au scénario dans le format demandé"
        +"\n Si certains prémisses sont des négations, utilise la caractère ~ au début de la chaîne. Par exemple:"
        +"\n 'Une ambulance avec son gyrophare n'a pas traversé le parc' donnerait:"
        +"\n véhicule;gyrophare;etat_urgence;~traverse_parc" 
        +"\n 'et, Une ambulance ne s'est pas arrêté au feu rouge' donnerait:"
        +"\n véhicule;traverse_feu_rouge" 
        +"\n Ton retour sera utilisé dans le contexte de textes juridiques. Adapte tes réponses à ce contexte."
        +"\n Attention, veille à créer des catégories qui font sens, un vélo peut être vu comme un véhicule "
        +"mais pas dans le contexte de la règle qui interdit aux véhicules motorisés de traverser un parc "
        +"\n Attention!! ne renvoie que un string de la forme demandée, pas d'explications!!!")
    elif lang == "en":
        prompt = f"""
                    System: You are an information extractor for legal scenarios.
                    Task: Extract ONLY atomic factual premises explicitly stated in the scenario text.
                    - No world knowledge, no guessing, no policy/judgment.
                    - You MAY reuse items from MEMORY only if they are explicitly stated in the scenario. MEMORY is not evidence.
                    - Normative/status terms (e.g., forbidden, allowed, permitted, prohibited, legal, illegal, authorized, required, must, may, should) are banned UNLESS an explicit normative word appears in the scenario (e.g., “forbidden”, “prohibited”, “authorized”, “permitted”, etc.).

                    Formatting:
                    - Output a SINGLE line: semicolon-separated snake_case tokens.
                    - Use "~" ONLY for explicit negation (when the text says something is NOT present/does NOT happen).
                    - No explanations, no extra text.

                    MEMORY (allowed tokens you MAY reuse if explicitly mentioned in the scenario):
                    {premises}

                    Examples (DO and DON’T):
                    - Scenario: "An ambulance drives through the park."
                    DO: vehicle;cross_park
                    DON'T: vehicle;cross_park;~forbidden   # normative inference, not stated

                    - Scenario: "It is forbidden for vehicles to cross the park."
                    DO: vehicle;cross_park;forbidden

                    - Scenario: "An ambulance didn't stop at the red light."
                    DO: vehicle;cross_red_light

                    Now decompose the following Scenario as a list of premises:
                    {scenario}

                    Return ONLY:
                    your_semicolon_list
                    """
    return prompt

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def get_complement(S_join,complement,session):              # returns a prompt complement in the language selected
    lang = session.get("lang", "fr")
    if lang == "fr":
        complement = ("Voici la décomposition que tu as proposé à l'étape précédente:"+S_join
            +"Voici des précisions de l'utilisateur pour l'améliorer:"+complement+"recommence en les prenant en compte")
    elif lang == "en":
        complement = ("Here is the decomposition you gave at last step:"+S_join
            +"Here are precisions from the user to make it better:"+complement+"try again")
    return complement

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def reset_session(session,keep_keys=None,reset_rules_updates=False):                # resets session while saving the keys in the keep_keys list
    if reset_rules_updates == True:
        lang = session.get("lang", "fr")
        base = lang
        new_base_path = f"{base}/RB_updated.txt"
        shutil.copy("uploads/RB_working.txt", new_base_path)
    if keep_keys is None:
        keep_keys = []

    preserved = {k: session.get(k) for k in keep_keys if k in session}
    session.clear()
    session.update(preserved)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#