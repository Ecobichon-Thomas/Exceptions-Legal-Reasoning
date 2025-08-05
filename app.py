from flask import Flask, request, render_template, session, redirect
from mistralai import Mistral
from logic import *
from lib_logic2 import *
import os
from huggingface_hub import InferenceClient
from werkzeug.utils import secure_filename


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

# def init_RB():
#     Rb = Rule_Base()
#     with open("W.txt","r", encoding="utf-8") as inp:
#         W = list(inp.read().splitlines())
#     with open("P.txt", "r", encoding="utf-8") as f:
#         lignes = f.readlines()
#     P = [ligne.strip().split() for ligne in lignes if ligne.strip()]
#     with open("C.txt","r",encoding="utf-8") as f:
#         lignes = f.readlines()
#     C = [ligne.strip().split() for ligne in lignes if ligne.strip()]
#     Rb.add_W(W)
#     Rb.add_rules(P,C)
#     return Rb

# def init_RB():
#     Rb = Rule_Base()
#     selected_w = session.get("selected_w", "W test")
#     selected_rb = session.get("selected_rb", "RB test")

#     with open(W_files[selected_w], "r", encoding="utf-8") as inp:
#         W = list(inp.read().splitlines())
#     with open(RB_files[selected_rb][0], "r", encoding="utf-8") as f:
#         P = [line.strip().split() for line in f if line.strip()]
#     with open(RB_files[selected_rb][1], "r", encoding="utf-8") as f:
#         C = [line.strip().split() for line in f if line.strip()]

#     Rb.add_W(W)
#     Rb.add_rules(P, C)
#     return Rb

def init_RB():
    Rb = Rule_Base()
    
    w_path = session.get("selected_w_path", "W.txt")
    p_path = session.get("selected_p_path", "P.txt")
    c_path = session.get("selected_c_path", "C.txt")

    with open(w_path, "r", encoding="utf-8") as inp:
        W = list(inp.read().splitlines())
    with open(p_path, "r", encoding="utf-8") as f:
        lignes = f.readlines()
    P = [ligne.strip().split() for ligne in lignes if ligne.strip()]
    with open(c_path, "r", encoding="utf-8") as f:
        lignes = f.readlines()
    C = [ligne.strip().split() for ligne in lignes if ligne.strip()]
    
    Rb.add_W(W)
    Rb.add_rules(P, C)
    return Rb

@app.context_processor
def inject_globals():
    return {
        "selected_api": session.get("selected_api", "HuggingFace"),
        "API": list(API_clients.keys()),
        "API_map": API_clients,
        "distances": list(DISTANCE_METHODS.keys()),
        "selection": list(SELECTION_METHODS.keys()),
        "dist_map": DISTANCE_METHODS,
        "sel_map": SELECTION_METHODS,
        "W_files": W_files,
        "RB_files": RB_files,
    }

#----------------------------------------------------------------------------------------------#

DISTANCE_METHODS = {"Distance de Hamming": "dist_hamming"}

SELECTION_METHODS = {"Seuil": "select_fct_treshold",
                     "Minimale":"select_fct_minimal",
                     "Seuil Minimal":"select_fct_treshold_minimal"}

API_clients = {"HuggingFace":client2,
               "Mistral":client1}

W_files = {"W test":"W.txt"}

RB_files = {"RB test":["P.txt","C.txt"]}


@app.route("/")
def index():
    return render_template(
        "Application.html",
        selected_api=session.get("selected_api", "HuggingFace"),
        API=list(API_clients.keys()),
        API_map=API_clients,
        distances=list(DISTANCE_METHODS.keys()),
        selection=list(SELECTION_METHODS.keys()),
        dist_map=DISTANCE_METHODS,
        sel_map=SELECTION_METHODS,
        W_files=W_files,
        RB_files=RB_files)

@app.route("/start")
def start():
    session["user_id"] = 42
    session["score"] = 0
    return "Session démarrée."

@app.route("/reset")
def reset():
    session.clear()
    return redirect("/")

@app.route("/set_config", methods=["POST"])
def set_config():
    session["selected_w"] = request.form.get("w_choice")
    session["selected_rb"] = request.form.get("rb_choice")
    session["selected_api"] = request.form.get("api")
    return redirect(request.referrer or url_for("index"))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload_rules", methods=["POST"])
def upload_rules():
    file_w = request.files.get("file_w")
    file_p = request.files.get("file_p")
    file_c = request.files.get("file_c")

    if not (file_w and file_p and file_c):
        return "Tous les fichiers sont requis", 400

    w_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_w.filename))
    p_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_p.filename))
    c_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_c.filename))

    file_w.save(w_path)
    file_p.save(p_path)
    file_c.save(c_path)

    session["selected_w_path"] = w_path
    session["selected_p_path"] = p_path
    session["selected_c_path"] = c_path

    return redirect(url_for("index"))

@app.route("/traiter", methods=["POST"])
def traiter():
    scenario = request.form.get('scenario', "").strip()             # Scenario dans le système, déjà décomposé
    resultat = request.form.get("resultat", "")             # Décomposition de ce scénario
    scenario1 = request.form.get("scenario1", "").strip()           # Nouveau scénario si l'utilisateur en a fourni un
    complement = request.form.get("complement", None)         #Complément au prompt si l'utilisateur n'est pas satisfait par la décomposition

    if scenario1 != "":
        selected_api = session.get("selected_api", "HuggingFace")  # sauvegarde le choix actuel
        session.clear()
        session["selected_api"] = selected_api  # restaure le choix après reset
        scenario = scenario1
    if scenario == "":              # Cas scénario vide
        return render_template("Application.html", 
                               resultat="Aucun scénario fourni.",
                               selected_api=session.get("selected_api", "HuggingFace"),
                                API=list(API_clients.keys()),
                                API_map=API_clients,
                                distances=list(DISTANCE_METHODS.keys()),
                                selection=list(SELECTION_METHODS.keys()),
                                dist_map=DISTANCE_METHODS,
                                sel_map=SELECTION_METHODS,
                                selected_w=session.get("selected_w", "W test"),
                                selected_rb=session.get("selected_rb", "RB test"),
                                W_files=W_files,
                                RB_files=RB_files)
        
    choice = request.form.get("user_choice", None)              # On récupère le choix de l'utilisateur si il y en a un (choix de quelle règle appliquer)
    Rb = init_RB()              # Initialisation de la base de règles

    if not session.get("decomposition") or (complement != None):                # Si on a la decomposition de S en mémoire ou que la décomposition n'est pas satisfaisante
        premises = ';'.join(Rb.Var_dictionnary._variables.keys())
        prompt = ("Tu es un expert des textes juridiques et de la décomposition de situations juridiques en prémices"
        +"Décompose le scénario sous la forme d'une liste de prémices, en utilisant le caractère ; comme séparateur dans ton retour."
        +"Voici un exemple de scénario, 'Une voiture a traversé un feu rouge', le résultat attendu serait, 'véhicule;traverse_feu_rouge': \n " 
        +"Scénario: \n"+ scenario 
        +"\n Liste des prémices déjà en mémoire qui peuvent être réutilisés si le scénario comporte des éléments similaires:"
        +premises
        +"\n Ne crée de nouveau prémice que si c'est nécessaire."
        +"\n Ne rajoute des prémices que lorsque tu as de l'information explicite, ne fait pas d'inférence. "
        +"\n Par exemple une ambulance n'est pas forcément en état d'urgence et n'a PAS FORCEMENT son gyrophare allumé! C'est le cas uniquement si c'est PRECISE, généralise cet exemple à tout les prémices"
        +"\n Ton retour ne doit comporter que la liste des prémices correspondant au scénario dans le format demandé"
        +"\n Si certains prémices sont des négations, utilise la caractère ~ au début de la chaîne. Par exemple:"
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
        resultat = request.form.get("resultat", "")
        S = session.get("decomposition", [])
        log = request.form.get("log", "")

    deja_appliquees = session.get("appliquees", [])

    if (choice is not None) and (choice != "-1"):                 # Si l'utilisateur a choisi
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

        log += "\n \n"+f"On applique la règle {choice} : {Rb.rules_original[choice]} "

        deja_appliquees.append(choice)
        session["appliquees"] = deja_appliquees

    analyse = scenario_check_web4_test(S, Rb,deja_appliquees)                # Règles applicables
    indices = analyse.get("indices",[])              # Indices des règles en question
    options = analyse.get("options",[])
    output = analyse.get("output","")

    if len(indices)==0 or choice=="-1":              # Plus aucune règle applicable, on s'arrête de générer l'extension
        if len(deja_appliquees) == 0:               # Si on a appliqué aucune règle on renvoie une erreur
            return render_template("Application.html",
                                   API=list(API_clients.keys()), 
                                    selected_api=session.get("selected_api", "HuggingFace"),
                                   resultat=resultat,
                                   scenario=scenario,
                                   No_rule = True,
                                   selected_w=session.get("selected_w", "W test"),
                                   selected_rb=session.get("selected_rb", "RB test"),
                                    W_files=W_files,
                                    RB_files=RB_files)
        
        else:               # Sinon on passe à la génération des exceptions
            log +="\n \n Il n'y a plus de règles applicables: Fin de la génération de l'extension"
            return render_template("Application.html",
                                   API=list(API_clients.keys()),
                                   selected_api=session.get("selected_api", "HuggingFace"),
                                scenario=scenario,
                                resultat=resultat,
                                extension = True,
                                log=log,
                                distances=list(DISTANCE_METHODS.keys()),
                                selection=list(SELECTION_METHODS.keys()),
                                dist_map=DISTANCE_METHODS,
                                sel_map=SELECTION_METHODS,
                                selected_w=session.get("selected_w", "W test"),
                                selected_rb=session.get("selected_rb", "RB test"),
                                W_files=W_files,
                                RB_files=RB_files)
        
    else:               # Si plusieurs règles possibles, on transmet les choix
        log +="\n".join(output)
        return render_template("Application.html",
                               API=list(API_clients.keys()),
                               selected_api=session.get("selected_api", "HuggingFace"),
                                conflit=True,
                                resultat=resultat,
                                options=options,
                                indices=indices,
                                scenario=scenario,
                                log=log,
                                selected_w=session.get("selected_w", "W test"),
                                selected_rb=session.get("selected_rb", "RB test"),
                                W_files=W_files,
                                RB_files=RB_files)

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
        return render_template(
        "Application.html",
        resultat=resultat,
        scenario = scenario,
        no_exception = True,
        API=list(API_clients.keys()),
        selected_api=session.get("selected_api", "HuggingFace"),
        distances=list(DISTANCE_METHODS.keys()),
        selection=list(SELECTION_METHODS.keys()),
        dist_map=DISTANCE_METHODS,
        sel_map=SELECTION_METHODS,
        log=log,
        selected_w=session.get("selected_w", "W test"),
        selected_rb=session.get("selected_rb", "RB test"),
        W_files=W_files,
        RB_files=RB_files)
    
    exceptions_string = choix_ex["regles_adaptees"].copy()
    for i,regle in enumerate(choix_ex["regles_adaptees"]):
        for j,exception in enumerate(regle) :
            P,C = exception
            P_str = ' ^ '.join(str(s) for s in P)
            C_str = ' ^ '.join(str(c) for c in C)
            exceptions_string[i][j] = (f"{P_str} => {C_str}")


    return render_template(
        "Application.html",
        API=list(API_clients.keys()),
        selected_api=session.get("selected_api", "HuggingFace"),
        resultat=resultat,
        scenario = scenario,
        options_rules = choix_ex["options"],
        indices_rules = choix_ex["indices"],
        options_exceptions = choix_ex["exceptions associées"],
        exceptions_string = exceptions_string,
        distances=list(DISTANCE_METHODS.keys()),
        selection=list(SELECTION_METHODS.keys()),
        dist_map=DISTANCE_METHODS,
        sel_map=SELECTION_METHODS,
        log=log,
        selected_w=session.get("selected_w", "W test"),
        selected_rb=session.get("selected_rb", "RB test"),
        W_files=W_files,
        RB_files=RB_files)

@app.route("/proposition", methods=["POST"])
def proposition():

    scenario = request.form.get('scenario', "").strip()
    log = request.form.get("log", "")

    regle_exception = request.form.get("choix_regle_exception", "")

    resultat = request.form.get("resultat", "")
    S = resultat.split(";")
    Rb = init_RB()
    Rb.init_S(S)

    return render_template(
        "Application.html",
        API=list(API_clients.keys()),
        selected_api=session.get("selected_api", "HuggingFace"),
        regle_exception = regle_exception,
        resultat=resultat,
        scenario = scenario,
        log=log,
        selected_w=session.get("selected_w", "W test"),
        selected_rb=session.get("selected_rb", "RB test"),
        W_files=W_files,
        RB_files=RB_files)

if __name__ == "__main__":
    app.run(debug=True)

#----------------------------------------------------------------------------------------------#
