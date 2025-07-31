import unittest
from lib_logic2 import *

class TestFcts(unittest.TestCase):

    def setUp(self):
        self.Var_dict = VariableUnicity()

    def test_get(self):
        var1 = self.Var_dict.get("var")
        var2 = self.Var_dict.get("var")
        self.assertIsInstance(var1, Variable)
        self.assertIs(var1,var2)
    
    def test_get_many(self):
        vars = self.Var_dict.get_many(*["var","var","a","b"])
        for v in vars:
            self.assertIsInstance(v, Variable)
        self.assertIs(vars[0],vars[1])

    def test_list_to_vars(self):
        vars = list_to_vars(self.Var_dict,["interdit","~authorisé"])
        self.assertIsInstance(vars[1],Not)
        self.assertIsInstance(vars[1].children[0],Variable)
        self.assertIsInstance(vars[0],Variable)

    def test_is_a_in_b(self):
        v1 = list_to_vars(self.Var_dict,["~a","~d"])
        v2 = list_to_vars(self.Var_dict,["~a","~d","c"])
        self.assertEqual(True,is_a_in_b(v1, v2))

    def test_get_var_from_index(self):
        var_objs = list_to_vars(self.Var_dict, ["a","b","c"])
        token_to_var = np.array(list(zip([1,3,7], var_objs)), dtype=object)
        self.assertIs(get_var_from_index(token_to_var, 3),self.Var_dict.get("b"))

    def test_extract_subtokens(self):
        token,index = extract_subtokens(["a","(","b","(","(","c",")",")",")"], 1)
        self.assertEqual(index,8)
        self.assertEqual(token,['b', '(', '(', 'c', ')', ')'])

    def test_eval_subformula(self):
        var_objs = list_to_vars(self.Var_dict, ["a","b","c"])
        token_to_var = np.array(list(zip([0,2,5], var_objs)), dtype=object)
        formula,indice  = eval_subformula(["a","(","b","&","(","c",")",")"], token_to_var, 1, True)

        vars_comp = list_to_vars(self.Var_dict,["a","b","c"])
        comparaison = ~(vars_comp[1]&(vars_comp[2]))

        self.assertEqual(formula,comparaison)
        self.assertEqual(indice,7)
    
    def test_formula_creator(self):
        tokens = ["~","a","&","(","~","~","~","b","|","c",")"]
        var_objs = list_to_vars(self.Var_dict, ["a","b","c"])
        token_to_var = np.array(list(zip([1,7,9], var_objs)), dtype=object)
        formula = formula_creator(tokens, token_to_var)

        vars_comp = list_to_vars(self.Var_dict,["a","b","c"])
        comparaison = ~vars_comp[0]&(~vars_comp[1]|(vars_comp[2]))

        self.assertEqual(formula,comparaison)

    def test_str_to_formula(self):
        Rb = Rule_Base()
        formula = str_to_formula("~ ( a & ~ ~ b ) <=> ~ ~ ~ ( a | ( ~ c & d ) | e )", Rb)

        vars_comp = list_to_vars(self.Var_dict,["a","b","c","d","e"])
        comparaison = (~(vars_comp[0]&vars_comp[1])).iff( ~(vars_comp[0]|(~vars_comp[2]&vars_comp[3])|vars_comp[4] ))

        self.assertEqual(formula,comparaison)

    def test_children_extraction(self):
        Rb = Rule_Base()
        formula = str_to_formula("~ ( a & ~ b ) <=> ~ ( c | d & ( e & f ) )", Rb)
        vars_comp = list_to_vars(Rb.Var_dictionnary,["a","b","c","d","e","f"])
        temp = children_extraction(formula)
        self.assertEqual(vars_comp,temp)

class TestRuleBase(unittest.TestCase):

    def setUp(self):
        self.Rb = Rule_Base()

    def test_add_rules_W_and_inclusion(self):
        self.Rb.add_W(["a <=> ~ b",
                       "~ c <=> d",
                       "~ ( e & f ) & ~ ( e & g ) & ~ ( g & f )",
                       "~ h >> i",
                       "j << ~ k"])
        self.Rb.add_rules([["b","~c"],
                           ["~a","d","f"], 
                           ["~j","g","d","f","b"], 
                           ["a","d","f"],
                           ["d","f"]],
                          [["h"] , ["i"] , ["h"] , ["e"] , ["~g"]])
        self.Rb.init_S(["b", "~c", "f","k"])
        included = self.Rb.inclusion([])                # On teste quelles règles peuvent s'appliquer à la situation S, on attend [0,1,3] comme résultat
        self.assertEqual([0,1,4], included)

    def test_init_S(self):
        self.Rb.add_W(["a >> ~ h",
                      "c <=> d"])
        self.Rb.init_S(["a","~b","c"])
        self.assertEqual(self.Rb.S_original,"a ^ ~b ^ c")
        self.assertEqual(self.Rb.S,list_to_vars(self.Rb.Var_dictionnary,["a","~b","c","~h","d"]))

    def test_dictionnaire_eval_rules (self):
        self.Rb.add_W(["f >> ~ h",
                      "~ ( b & f ) & ~ ( b & d ) & ~ ( d & f )"])
        self.Rb.add_rules([["a"],
                           ["a","c","e"],
                           ["a","c"],
                           ["a","c","e","g"]],
                          [["b"] , ["f"] , ["d"] , ["b"]])
        self.Rb.init_S(["a","c","e"])
        rules = [self.Rb.rules[0],self.Rb.rules[1]]

        TD1,Prop1 = dictionnaire_eval_rules (self.Rb,rules,conclusions_only = False)
        self.assertEqual(TD1, {"a": True, "b": True, "c": True, "d": False, "e": True, "f": True, "g": False, "h": False})
        self.assertEqual(Prop1, ["a", "c", "e", "b", "f","h"])

        TD2,Prop2 = dictionnaire_eval_rules (self.Rb,rules,conclusions_only = True)
        self.assertEqual(TD2, {"a": False, "b": True, "c": False, "d": False, "e": False, "f": True, "g": False, "h": False})
        self.assertEqual(Prop2, ["b", "f", "h"])

        TD3,Prop3 = dictionnaire_eval_rules (self.Rb,[self.Rb.rules[2]],conclusions_only = False)
        self.assertEqual(TD3, {"a": True, "b": False, "c": True, "d": True, "e": True, "f": False, "g": False, "h": False})
        self.assertEqual(Prop3, ["a", "c", "e","d"])

        TD4,Prop4 = dictionnaire_eval_rules (self.Rb,[self.Rb.rules[2]],conclusions_only = True)
        self.assertEqual(TD4, {"a": False, "b": False, "c": False, "d": True, "e": False, "f": False, "g": False, "h": False})
        self.assertEqual(Prop4, ["d"])

    def test_compatible(self):
        self.Rb.add_W(["f >> ~ h",
                      "~ ( b & f ) & ~ ( b & d ) & ~ ( d & f )"])
        self.Rb.add_rules([["a"],
                           ["a","c","e"],
                           ["a","c"],
                           ["a","c","e","g"]],
                          [["b"] , ["f"] , ["d"] , ["b"]])
        self.Rb.init_S(["a","c","e"])

        rules1 = [self.Rb.rules[0],self.Rb.rules[1]]
        self.assertEqual(False,
                         self.Rb.compatible(rules1,conclusions_only=False))

        rules2 = [self.Rb.rules[0],self.Rb.rules[2]]
        self.assertEqual(False,
                         self.Rb.compatible(rules2,conclusions_only=False))

        rules3 = [self.Rb.rules[0],self.Rb.rules[3]]
        self.assertEqual(True,
                         self.Rb.compatible(rules3,conclusions_only=False))
        
        rules4 = [self.Rb.rules[0],self.Rb.rules[3]]
        self.assertEqual(True, 
                         self.Rb.compatible(rules4,conclusions_only=False))


    def test_compatibility_matrix(self):
        self.Rb.add_W(["f >> ~ h",
                       "~ ( b & f ) & ~ ( b & d ) & ~ ( d & f )"])
        self.Rb.add_rules([["a"],
                           ["a","c","e"],
                           ["a","c"],
                           ["a","c","e","g"]],
                          [["b"] , ["f"] , ["d"] , ["h"]])
        self.Rb.init_S(["b", "~c", "f","k"])
        mat = self.Rb.compatibility_matrix([0,1,2,3])
        expected = [[0,1,1,0],
            [0,0,0,1],
            [0,1,0,0],
            [0,0,0,0]]                # règle 4 incompatible avec 2,3,4 car sa ccl incompatible avec f qui est un des pré-requis des autres

        for row1, row2 in zip(mat, expected):
            self.assertEqual(row1.tolist(), row2)     

    def test_dist_hamming(self):
        self.Rb.add_W(["f >> ~ h",
                      "~ ( b & f ) & ~ ( b & d ) & ~ ( d & f )"])
        self.Rb.add_rules([["a"],
                           ["a","c","e"],
                           ["a","c"],
                           ["a","c","e","g"]],
                          [["b"] , ["f"] , ["d"] , ["b"]])
        dists = self.Rb.dist_hamming(0)
        self.assertEqual(dists, [0,9,9,3])  # identiques

    def test_ensemble_premices_equi(self):
        self.Rb.add_W(["f >> ~ h", 
                       "~ ( b & f ) & ~ ( b & d ) & ~ ( d & f )",
                       "a <=> e",
                       "e >> c",
                       "~ j << c"])
        temp = ensemble_premices_equi(list_to_vars(self.Rb.Var_dictionnary,["f","a"]),self.Rb.W)
        comparaison = list_to_vars(self.Rb.Var_dictionnary,["f","a","~h","e","c","~j"])

        self.assertIs(temp[0],comparaison[0])

        self.assertIs(temp[1],comparaison[1])

        self.assertIsInstance(temp[2],Not)
        self.assertIs(temp[2].children[0],comparaison[2].children[0])

        self.assertIs(temp[3],comparaison[3])

        self.assertIs(temp[4],comparaison[4])

        self.assertIsInstance(temp[5],Not)
        self.assertIs(temp[5].children[0],comparaison[5].children[0])

class TestSelection(unittest.TestCase):

    def test_select_fct_treshold (self):
        temp = select_fct_treshold([10,0,3,7,5,2,10,3],6)
        for i,t in enumerate([2,4,5,7]):
            self.assertEqual(t,temp[i])

    def test_select_fct_minimal (self):
        temp = select_fct_minimal([10,0,3,7,5,2,10,3])
        self.assertEqual(5,temp[0])

        temp = select_fct_minimal([10,0,3,7,5,4,10,3])
        self.assertEqual(2,temp[0])
        self.assertEqual(7,temp[1])

    def test_select_fct_treshold_minimal (self):
        temp = select_fct_treshold_minimal([10,0,3,7,5,2,10,3],6)
        self.assertEqual(5,temp[0])

        temp = select_fct_treshold_minimal([10,0,3,7,5,2,10,3],1)
        self.assertEqual(temp,[])
if __name__ == '__main__':
    unittest.main()