from flask import Flask, request, render_template, session, redirect, url_for
from flask_babel import Babel
from flask_babel import lazy_gettext as _
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

#---------------------------------------------Babel-------------------------------------------------#

app.config['BABEL_DEFAULT_LOCALE'] = 'fr'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'
app.config['BABEL_SUPPORTED_LOCALES'] = ['fr', 'en']

def get_locale():
    return session.get('lang', 'fr')

babel = Babel(app, locale_selector=get_locale)

#---------------------------------------------Extraction des clés-------------------------------------------------#

with open("cles_API.txt") as inp:
    keys = list(inp.read().split())

#------------------------------------------------------------APIs------------------------------------------------------------#

MODELS = ["mistral-medium","meta-llama/Llama-3.3-70B-Instruct"]
clients = [Mistral(api_key=keys[0]),InferenceClient(provider="hyperbolic",api_key=keys[1])]

#------------------------------------------------------------Autres vars------------------------------------------------------------#

API_clients = {"Ollama":clients[0],
               "Mistral":clients[1]}

html_file = "Application.html"

#------------------------------------------------------------------------------------------------------------------------#

session_cleared = False

@app.before_request
def clear_session_once_per_server():
    global session_cleared

    if not session_cleared: 
        if os.path.isfile("en/RB/RB_updated.txt"):
            os.remove("en/RB/RB_updated.txt")
        if os.path.isfile("fr/RB/RB_updated.txt"):
            os.remove("fr/RB/RB_updated.txt")
        session.clear()
        session_cleared = True

#------------------------------------------------------------------------------------------------------------------------#

@app.route('/set_language/<lang>')
def set_language(lang):
    reset_session(session,
                ["lang",
                "selected_api",])
    session['lang'] = lang
    return redirect(url_for('index'))

#------------------------------------------------------------------------------------------------------------------------#

@app.context_processor
def inject_globals():
    default_w_path, default_rb_path, W_files, RB_files = get_files(session)
    DISTANCE_METHODS = get_distance_method()
    SELECTION_METHODS = get_selection_method()
    context = {
        "selected_api": session.get("selected_api", "Ollama"),
        "API": list(API_clients.keys()),
        "API_map": API_clients,
        "distances": list(DISTANCE_METHODS.keys()),
        "selection": list(SELECTION_METHODS.keys()),
        "dist_map": DISTANCE_METHODS,
        "sel_map": SELECTION_METHODS,
        "W_files": W_files,
        "RB_files": RB_files,
        "selected_w": session.get("selected_w", "W_test"),
        "selected_rb": session.get("selected_rb", "RB_test"),
    }
    context.update({k: session[k] for k in ["scenario", "resultat", "log"] if k in session})
    return context

#----------------------------------------------------------------------------------------------#


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
    default_w_path, default_rb_path, W_files, RB_files = get_files(session)
    session["upload_init"] = True
    w_choice = request.form.get("w_choice")
    rb_choice = request.form.get("rb_choice")

    session["selected_w"] = w_choice

    if w_choice == "__upload__" :
        file_w = request.files.get("uploaded_w")
        if not file_w:
            return "Tous les fichiers sont requis", 400
        
        w_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_w.filename))
        file_w.save(w_path)

        session["uploaded_w_filename"] = file_w.filename
        session["selected_w_path"] = w_path
    else :
        if w_choice not in W_files:
            return "Fichiers prédéfinis invalides", 400
        
        w_original_path = W_files[w_choice]

        session["uploaded_w_filename"] = None
        session["selected_w_path"] = w_original_path

    session["selected_rb"] = rb_choice
    session["rb"] = rb_choice
    if rb_choice == "__upload__":
        session["rb"] = file_rb.filename
        file_rb = request.files.get("uploaded_rb")
        if not file_rb:
            return "Tous les fichiers sont requis", 400
        rb_original_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file_rb.filename))
        file_rb.save(rb_original_path)

        rb_working_path = os.path.join(app.config["UPLOAD_FOLDER"], "RB_working.txt")
        shutil.copy(rb_original_path, rb_working_path)

        session["uploaded_rb_filename"] = file_rb.filename
        session["original_rb_path"] = rb_original_path
        session["selected_rb_path"] = rb_working_path
    else:
        if rb_choice not in RB_files:
            return "Fichiers prédéfinis invalides", 400
        
        session["uploaded_rb_filename"] = None
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
        default_w_path, default_rb_path, W_files, RB_files = get_files(session)
        shutil.copy(session.get("original_rb_path", default_rb_path), os.path.join(app.config["UPLOAD_FOLDER"], "RB_working.txt"))              # On ré-initialise le fichier de travail RB_working
        reset_session(session,
                      ["lang",
                       "selected_api",
                       "upload_init",
                       "selected_w_path",
                       "selected_rb_path",
                       "selected_w",
                       "uploaded_w_filename",
                       "selected_rb",
                       "uploaded_rb_filename",
                       "rb"])
        scenario = scenario1
    if scenario == "":              # Cas scénario vide
        session["resultat"] = get_log(1)
        return render_template(html_file)
        
    choice = request.form.get("user_choice", None)              # On récupère le choix de l'utilisateur si il y en a un (choix de quelle règle appliquer)
    Rb = init_RB(session)              # Initialisation de la base de règles

    if not session.get("decomposition") or (complement != None):                # Si on a la decomposition de S en mémoire ou que la décomposition n'est pas satisfaisante
        premises = ';'.join(Rb.Var_dictionnary._variables.keys())
        prompt = get_prompt(scenario,premises,session)

        if (complement != None):
            S_join = ';'.join(session.get("decomposition"))
            prompt += get_complement(S_join,complement,session)
        
        resultat = call_llm(prompt,MODELS,clients,session)
    
        S = resultat.split(";")

        session["decomposition"] = S
        session["appliquees"] = []
        premier_log = True               # Pour le début du log comme la traduction ne marche pas dans app.py
        log = ""
    else :
        S = session.get("decomposition", [])
        log = request.form.get("log") or request.args.get("log") or ""
        premier_log = False

    deja_appliquees = session.get("appliquees", [])

    if (choice is not None) and (choice != "-1") and (choice != "-2") and (choice != "-3"):                 # Si l'utilisateur a choisi
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

        log += "\n\n"
        log += "\n \n"+f" {get_log(2)} {choice} : {Rb.rules_original[choice]} "

        deja_appliquees.append(choice)
        session["appliquees"] = deja_appliquees

    analyse = scenario_check_web4_test(S, Rb,deja_appliquees,premier_log)                # Règles applicables
    indices = analyse.get("indices",[])              # Indices des règles en question
    options = analyse.get("options",[])
    output = analyse.get("output","")

    session["resultat"]=resultat
    session["scenario"]=scenario

    if len(indices)==0 and len(deja_appliquees) == 0:              # Si aucune règle n'est applicable est aucune règle n'a été appliquée, message d'erreur
        return render_template(html_file,
                                No_rule = True,
                                log="No log")
    elif choice=="-1":              # génération d'une exception
            return render_template(html_file,
                                   extension = True,
                                   log=log)
    elif choice=="-2":              # recap
        new_rules = session.get("new_rules", False)
        if new_rules == True:
            return render_template(html_file,
                                recap = True,
                                add_RB=True,
                                regles_appliquees = [Rb.rules[i] for i in deja_appliquees],
                                situation_finale = [str(s) for s in S],
                                log=log)
        else:
            return render_template(html_file,
                    recap = True,
                    regles_appliquees = [Rb.rules[i] for i in deja_appliquees],
                    situation_finale = [str(s) for s in S],
                    log=log)
    elif choice=="-3":              # changement ou non de la base utilisée
        add_RB = request.form.get("add_RB", False)
        if add_RB == "True":
            i = 0
            while i<10:
                file_name = session.get("rb", "RB_test")
                base = session.get("lang", "fr")
                new_RB_path = f"{base}/RB/{file_name}_updated{str(i)}.txt"
                if not os.path.exists(new_RB_path):
                     shutil.copy("uploads/RB_working.txt", new_RB_path)
                     break
                i+=1

        return render_template(html_file,
                               recap=True,
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

#------------------------------------------------------------------------------------------------------------------------#

@app.route("/exceptions", methods=["POST"])
def exception():
    DISTANCE_METHODS = get_distance_method()
    SELECTION_METHODS = get_selection_method()
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
    Rb = init_RB(session)
    Rb.init_S(S)

    choix_ex = choix_exception(distance_method, Rb, arguments,deja_appliquees[-1])

    if choix_ex["options"] == []:
        log += choix_ex["output"]
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

#------------------------------------------------------------------------------------------------------------------------#

@app.route("/proposition", methods=["POST"])
def proposition():
    scenario = request.form.get('scenario', "").strip()
    log = request.form.get("log", "")

    choix = request.form.get("choix_exception")
    deja_appliquees = session["appliquees"]

    resultat = request.form.get("resultat", "")
    S = resultat.split(";")

    if choix == "-1":
        log +=f"\n \n {get_log(3)} "
        return redirect(url_for(
                "traiter",
                scenario=scenario,
                log=log,
                resultat=resultat
            ))
    else:
        session["new_rules"] = True
        i_str, j_str = choix.split("|")
        i = int(i_str)
        j = int(j_str)
        
        P,C = session.get("adaptations", [])[i][j]
        P_str = ' '.join(s.replace(" ", "") for s in P)
        C_str = ' '.join(c.replace(" ", "") for c in C)
        adaptation_string = (f"{P_str}    >   {C_str}")
        with open(session["selected_rb_path"], "a", encoding="utf-8") as f:
            f.write("\n"+ adaptation_string)

        Rb = init_RB(session)
        Rb.init_S(S)
        ancienne_conclu = Rb.rules[deja_appliquees[-1]].conclusion              # suppression de l'ancienne conclu
        nouvelle_decomposition = difference_premisses (session.get("decomposition", []),ancienne_conclu,Rb.W)
        session["decomposition"] = nouvelle_decomposition

        # comme on ajoute la nouvelle règle à la fin, son indice correspond au compteur
        log +=f"\n \n {get_log(4)} : {Rb.rules[Rb.compteur-1]} \n {get_log(5)} : {Rb.rules[deja_appliquees[-1]]}"
        log +=f"\n \n {get_log(6)} : {Rb.rules[deja_appliquees[-1]]} \n {get_log(7)} "
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
