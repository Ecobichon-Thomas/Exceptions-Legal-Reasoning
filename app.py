from flask import Flask, request, render_template, session, redirect, url_for
from mistralai import Mistral
from logic import *
from lib_logic2 import *
import os
from huggingface_hub import InferenceClient
from werkzeug.utils import secure_filename
import shutil
import copy

app = Flask(__name__)
app.secret_key = "test75591729"

#---------------------------------------------Extraction des clés-------------------------------------------------#

with open("cles_API.txt") as inp:
    keys = list(inp.read().split())

#------------------------------------------------------------MISTRAL------------------------------------------------------------#
API_KEY = keys[0]
MODEL1 = "mistral-medium"
client1 = Mistral(api_key=API_KEY)

#------------------------------------------------------------HUGGINGFACE_OLLAMA------------------------------------------------------------#
MODEL2 = "meta-llama/Llama-3.3-70B-Instruct"
os.environ["HF_TOKEN"] = keys[1]
client2 = InferenceClient(provider="hyperbolic",api_key=os.environ["HF_TOKEN"])

def call_llm(prompt):
    api_order = [session.get("selected_api", "HuggingFace"), "Mistral", "HuggingFace"]
    tried = set()

    for api_name in api_order:
        if api_name in tried:
            continue
        tried.add(api_name)

        try:
            if api_name == "Mistral":
                response = client1.chat.complete(
                    model=MODEL1,
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                response = client2.chat.completions.create(
                    model=MODEL2,
                    messages=[{"role": "user", "content": prompt}]
                )

            session["selected_api"] = api_name
            return response.choices[0].message.content

        except Exception as e:
            if "capacity" in str(e).lower() or "rate limit" in str(e).lower():
                continue
            raise e

    # Si aucune API ne marche :
    raise RuntimeError("Aucune API disponible actuellement.")

def init_RB():
    Bool = session.get("upload_init", False)
    Rb = Rule_Base()

    w_path = session.get("selected_w_path", "W.txt")
    if not Bool:
        shutil.copy("RB.txt", "uploads/RB_working.txt")
        session["selected_rb_path"] = "uploads/RB_working.txt"
        session["original_rb_path"] = "RB.txt"
        session["upload_init"]=True
        session["selected_w_path"] = w_path

    rb_path = "uploads/RB_working.txt"

    P=[]
    C=[]

    with open(rb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # Ignore lignes vides ou commentaires
                continue
            if ">" in line:
                gauche, droite = line.split(">", 1)  # Split en deux parties seulement
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

@app.context_processor
def inject_globals():
    context = {
        "selected_api": session.get("selected_api", "HuggingFace"),
        "API": list(API_clients.keys()),
        "API_map": API_clients,
        "distances": list(DISTANCE_METHODS.keys()),
        "selection": list(SELECTION_METHODS.keys()),
        "dist_map": DISTANCE_METHODS,
        "sel_map": SELECTION_METHODS,
        "W_files": W_files,
        "RB_files": RB_files,
        "selected_w": session.get("selected_w", "W test"),
        "selected_rb": session.get("selected_rb", "RB test"),
    }

    context.update({k: session[k] for k in ["scenario", "resultat", "log"] if k in session})

    return context
#----------------------------------------------------------------------------------------------#

DISTANCE_METHODS = {"Distance de Hamming": "dist_hamming"}

SELECTION_METHODS = {"Seuil": "select_fct_treshold",
                     "Minimale":"select_fct_minimal",
                     "Seuil Minimal":"select_fct_treshold_minimal"}

API_clients = {"HuggingFace":client2,
               "Mistral":client1}

W_files = {"W test":"W.txt"}

RB_files = {"RB test":"RB.txt"}

html_file = "Application.html"

session_cleared = False
@app.before_request
def clear_session_once_per_server():
    global session_cleared

    if not session_cleared:
        session.clear()
        session_cleared = True


@app.route("/")
def index():
    return render_template(html_file)

# @app.route("/start")
# def start():
#     session["user_id"] = 42
#     session["score"] = 0
#     return "Session démarrée."

@app.route("/reset")
def reset():
    session.clear()
    return redirect("/")

#----------------------------------------------------------------------------------------------#

@app.route("/set_config", methods=["POST"])             # Utilisé quand on donne des fichiers locaux
def set_config():
    session["selected_api"] = request.form.get("api")
    return redirect(request.referrer or url_for("index"))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload_rules", methods=["POST"])
def upload_rules():
    session["upload_init"] = True
    w_choice = request.form.get("w_choice")
    rb_choice = request.form.get("rb_choice")

    if w_choice == "__upload__" :
        file_w = request.files.get("uploaded_w")
        if not file_w:                #FIX
            return "Tous les fichiers sont requis", 400
        w_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_w.filename))
        file_w.save(w_path)
        session["selected_w_path"] = w_path
    else :
        if w_choice not in W_files:
            return "Fichiers prédéfinis invalides", 400
        
        w_original_path = W_files[w_choice]
        session["selected_w_path"] = w_original_path


    if rb_choice == "__upload__":
        file_rb = request.files.get("uploaded_rb")
        if not file_rb:                #FIX
            return "Tous les fichiers sont requis", 400
        rb_original_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_rb.filename))
        file_rb.save(rb_original_path)

        rb_working_path = os.path.join(app.config["UPLOAD_FOLDER"], "RB_working.txt")
        shutil.copy(rb_original_path, rb_working_path)

        session["original_rb_path"] = rb_original_path
        session["selected_rb_path"] = rb_working_path
    else:
        if rb_choice not in RB_files:
            return "Fichiers prédéfinis invalides", 400
        
        rb_original_path = RB_files[rb_choice]
        rb_working_path = os.path.join(app.config["UPLOAD_FOLDER"], "RB_working.txt")
        shutil.copy(rb_original_path, rb_working_path)
        session["original_rb_path"] = rb_original_path
        session["selected_rb_path"] = rb_working_path

    return redirect(url_for("index"))

#----------------------------------------------------------------------------------------------#

@app.route("/traiter", methods=["GET","POST"])
def traiter():
    scenario = (request.form.get('scenario') or request.args.get('scenario') or "").strip()             # Scenario dans le système, déjà décomposé
    resultat = request.form.get("resultat") or request.args.get("resultat") or ""             # Décomposition de ce scénario
    scenario1 = (request.form.get("scenario1") or request.args.get("scenario1") or "").strip()           # Nouveau scénario si l'utilisateur en a fourni un
    complement = request.form.get("complement") or request.args.get("complement")         #Complément au prompt si l'utilisateur n'est pas satisfait par la décomposition

    if scenario1 != "":
        print("Nouveau scénario")
        selected_api = session.get("selected_api", "HuggingFace")  # sauvegarde le choix actuel
        upload_init = session.get("upload_init", False)
        print("upload_init",upload_init)
        print("session",session)
        try:
            rb_path = session["selected_rb_path"]
            print("rb_path",rb_path)
            print("original_rb_path",session["original_rb_path"])
            shutil.copy(session["original_rb_path"], session["selected_rb_path"])
            w_path = session["selected_w_path"]
            print("w_path",w_path)
        except:
            None
        session.clear()
        print("CLEAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        session["selected_api"] = selected_api  # restaure le choix après reset
        session["upload_init"] = upload_init
        try :
            session["selected_w_path"] = w_path
            session["selected_rb_path"] = rb_path
        except :
            None
        scenario = scenario1
    if scenario == "":              # Cas scénario vide
        session["resultat"] = "Aucun scénario fourni."
        return render_template(html_file)
        
    choice = request.form.get("user_choice", None)              # On récupère le choix de l'utilisateur si il y en a un (choix de quelle règle appliquer)
    Rb = init_RB()              # Initialisation de la base de règles

    if not session.get("decomposition") or (complement != None):                # Si on a la decomposition de S en mémoire ou que la décomposition n'est pas satisfaisante
        premises = ';'.join(Rb.Var_dictionnary._variables.keys())
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

        if (complement != None):
            S_join = ';'.join(session.get("decomposition"))
            prompt += ("Voici la décomposition que tu as proposé à l'étape précédente:"+S_join
            +"Voici des précisions de l'utilisateur pour l'améliorer:"+complement+"recommence en les prenant en compte")
        
        resultat = call_llm(prompt)
    
        S = resultat.split(";")

        session["decomposition"] = S
        session["appliquees"] = []
        log = "Génération d'une extension:"
    else :
        S = session.get("decomposition", [])
        log = request.form.get("log") or request.args.get("log") or ""

    deja_appliquees = session.get("appliquees", [])

    if (choice is not None) and (choice != "-1") and (choice != "-2"):                 # Si l'utilisateur a choisi
        # on ajoute directement la règle puis on passe à la suite
        deja_appliquees = session.get("appliquees", [])

        choice = int(choice)
        ccl = Rb.rules[choice].conclusion              # robuste pour passage aux listes de conclusions
        for c in ccl:                   # On traduit en string les négations pour pouvoir les entrer dans S
            if isinstance(c,Not):
                temp = "~"+c.children[0].name
            else:
                temp = c.name
            if temp not in S:
                S.append(temp)

        log += "\n \n"+f"Application de la règle {choice} : {Rb.rules_original[choice]} "

        deja_appliquees.append(choice)
        session["appliquees"] = deja_appliquees

    analyse = scenario_check_web4_test(S, Rb,deja_appliquees)                # Règles applicables
    indices = analyse.get("indices",[])              # Indices des règles en question
    options = analyse.get("options",[])
    output = analyse.get("output","")

    session["resultat"]=resultat
    session["scenario"]=scenario

    if len(indices)==0 and len(deja_appliquees) == 0:              # Si aucune règle n'est applicable est aucune règle n'a été appliquée, message d'erreur
        return render_template(html_file,
                                No_rule = True)
    elif choice=="-1":              # génération d'une exception
            return render_template(html_file,
                                   extension = True,
                                   log=log)
    elif choice=="-2":              # recap
        return render_template(html_file,
                               recap = True,
                               regles_appliquees = [Rb.rules[i] for i in deja_appliquees],
                               situation_finale = [str(s) for s in S],
                               log=log)
    else:               # Si règles possibles, on transmet les choix 
        # (si aucune règle possible on pourra alors demander un recap, ou chercher une exception à la dernière règle appliquée)
        log +="\n".join(output)
        pas_premier = True
        if len(deja_appliquees)==0:
            pas_premier = False
        return render_template(html_file,
                                conflit=True,
                                options=options,
                                indices=indices,
                                log=log,
                                pas_premiere_regle=pas_premier)

@app.route("/exceptions", methods=["POST"])
def exception():
    dist_choice_label = request.form.get("distances", list(DISTANCE_METHODS.keys())[0]).strip()
    sel_choice_label = request.form.get("selection", list(SELECTION_METHODS.keys())[0])
    distance_method = DISTANCE_METHODS[dist_choice_label]
    selection_method = SELECTION_METHODS[sel_choice_label]

    scenario = request.form.get('scenario', "").strip()
    log = request.form.get("log", "")
    seuil = request.form.get("seuil")

    deja_appliquees = session["appliquees"]

    if seuil is not None:
        arguments = list([selection_method,seuil])
    else:
        arguments = list([selection_method])

    resultat = request.form.get("resultat", "")
    S = resultat.split(";")
    Rb = init_RB()
    Rb.init_S(S)

    choix_ex = choix_exception(distance_method, Rb, arguments,deja_appliquees[-1])

    if choix_ex["options"] == []:
        log +=f"\n \n Aucune des règles de la base n'est suffisamment proche pour proposer une adaptation d'exception"
        return render_template(
            html_file,
            Pas_adaptation = True,
            log = log)
    
    adaptations_string = copy.deepcopy(choix_ex["regles_adaptees"])
    adaptations_list = copy.deepcopy(choix_ex["regles_adaptees"])
    for i,regle in enumerate(choix_ex["regles_adaptees"]):
        for j,exception in enumerate(regle) :
            P,C = exception
            P_list = [str(v) for v in P]
            C_list = [str(v) for v in C]
            P_str = ' ^ '.join(str(s) for s in P)
            C_str = ' ^ '.join(str(c) for c in C)
            adaptations_string[i][j] = (f"{P_str} => {C_str}")
            adaptations_list[i][j] = [P_list,C_list]
    session["adaptations"]=adaptations_list

    return render_template(
        html_file,
        resultat=resultat,
        scenario = scenario,
        options_rules = choix_ex["options"],
        indices_rules = choix_ex["indices"],
        options_exceptions = choix_ex["exceptions associées"],
        adaptations_string = adaptations_string,
        log=log)

@app.route("/proposition", methods=["POST"])
def proposition():
    scenario = request.form.get('scenario', "").strip()
    log = request.form.get("log", "")

    choix = request.form.get("choix_exception")
    deja_appliquees = session["appliquees"]

    resultat = request.form.get("resultat", "")
    S = resultat.split(";")

    if choix == "-1":
        log +=f"\n \n Aucune des exceptions proposée n'a été validée"
        return redirect(url_for(
                "traiter",
                scenario=scenario,
                log=log,
                resultat=resultat
            ))
    else:
        i_str, j_str = choix.split("|")
        i = int(i_str)
        j = int(j_str)
        
        P,C = session.get("adaptations", [])[i][j]
        P_str = ' '.join(s.replace(" ", "") for s in P)
        C_str = ' '.join(c.replace(" ", "") for c in C)
        adaptation_string = (f"{P_str}    >   {C_str}")
        with open(session["selected_rb_path"], "a", encoding="utf-8") as f:
            f.write("\n"+ adaptation_string)

        Rb = init_RB()
        Rb.init_S(S)
        ancienne_conclu = Rb.rules[deja_appliquees[-1]].conclusion              # suppression de l'ancienne conclu
        nouvelle_decomposition = difference_premisses (session.get("decomposition", []),ancienne_conclu,Rb.W)
        session["decomposition"] = nouvelle_decomposition

        # comme on ajoute la nouvelle règle à la fin, son indice correspond au compteur
        log +=f"\n \n Création d'une exception: {Rb.rules[Rb.compteur-1]} \n à la règle: {Rb.rules[deja_appliquees[-1]]}"
        log +=f"\n \n Suite à la création réussie d'une exception à la règle: {Rb.rules[deja_appliquees[-1]]} \n on annule son application pour poursuivre la génération d'une extension"
        session["appliquees"] = deja_appliquees[:-1]                # Suppression de la dernière règle appliquée
    #regle_exception = request.form.get("choix_regle_exception", "")

    return redirect(url_for(
            "traiter",
            scenario=scenario,
            log=log,
            resultat=resultat
        ))

if __name__ == "__main__":
    app.run(debug=True)

#----------------------------------------------------------------------------------------------#
