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

# --------------------------------------------------
# Flask app initialization
# --------------------------------------------------

app = Flask(__name__)
app.secret_key = "test75591729"              # Secret key is needed for session handling (cookies, etc.)

#---------------------------------------------Babel-------------------------------------------------#

# Multi-languages, more explanatory details in comment in babel.cfg file

app.config['BABEL_DEFAULT_LOCALE'] = 'fr'                   # Default language = French
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'
app.config['BABEL_SUPPORTED_LOCALES'] = ['fr', 'en']

def get_locale():                    # Retrieve the language from the session (default = French)
    return session.get('lang', 'fr')

babel = Babel(app, locale_selector=get_locale)

#---------------------------------------------API keys extraction-------------------------------------------------#

with open("cles_API.txt") as inp:
    keys = list(inp.read().split())

keys = []
noms = []

with open("cles_API.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):  # Ignore empty lines or comments
            continue
        if ":" in line:
            gauche, droite = line.split(":", 1)
            nom = gauche.strip().split()
            key = droite.strip().split()

            keys.append(key[0])
            noms.append(nom[0])
        else:
            print("Ligne ignorée (pas de ':'):", line)

#------------------------------------------------------------API ilient init------------------------------------------------------------#

list_API = ["Mistral","Ollama"]
API_clients = {}
clients =[]
MODELS = []
first_api = ""
for i,nom in enumerate(noms):
    if nom == 'Mistral':
        if first_api != "Ollama":
            first_api = "Mistral"
        API_clients.update({"Mistral" : Mistral(api_key=keys[i])})
        MODELS.append("mistral-medium")
        clients.append(Mistral(api_key=keys[i]))
    elif nom == 'Ollama':
        first_api = "Ollama"
        API_clients.update({"Ollama" : InferenceClient(provider="hyperbolic",api_key=keys[i])})
        MODELS.append("meta-llama/Llama-3.3-70B-Instruct")
        clients.append(InferenceClient(provider="hyperbolic",api_key=keys[i]))

#------------------------------------------------------------Other vars------------------------------------------------------------#

html_file = "Application.html"

#------------------------------------------------------------------------------------------------------------------------#

session_cleared = False

@app.before_request
def clear_session_once_per_server():
    """
    This runs before each request.
    Used here to clear the session and reset RB_updated.txt once,
    the first time the server handles a request.
    """
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
    """
    Change the interface language.
    Flask route with a dynamic parameter <lang>.
    Example: /set_language/en or /set_language/fr
    """
    reset_session(session,
                ["lang",
                "selected_api",])
    session['lang'] = lang
    return redirect(url_for('index'))

#------------------------------------------------------------------------------------------------------------------------#

@app.context_processor
def inject_globals():
    """
    Add variables to the template context globally,
    so they can be accessed directly in HTML templates (Jinja2).
    """
    default_w_path, default_rb_path, W_files, RB_files = get_files(session)
    DISTANCE_METHODS = get_distance_method()
    SELECTION_METHODS = get_selection_method()
    context = {             # always defined
        "selected_api": session.get("selected_api", first_api),
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
    context.update({k: session[k] for k in ["scenario", "resultat", "log"] if k in session})            # if defined
    return context

#----------------------------------------------------------------------------------------------#


@app.route("/")
def index():
    """
    Root URL of the web app (homepage).
    Renders Application.html.
    """
    return render_template(html_file)

# @app.route("/start")
# def start():
#     session["user_id"] = 42
#     session["score"] = 0
#     return "Session démarrée."

@app.route("/reset")
def reset():
    """
    Reset the session and redirect to homepage.
    """
    session.clear()
    return redirect("/")

#----------------------------------------------------------------------------------------------#

@app.route("/set_config", methods=["POST"])
def set_config():
    """
    Route used when user selects configuration (e.g., choosing API).
    Accepts POST only.
    """
    session["selected_api"] = request.form.get("api")
    return redirect(request.referrer or url_for("index"))


UPLOAD_FOLDER = "uploads"               # creating a folder for Uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/upload_rules", methods=["POST"])
def upload_rules():
    """
    Route for uploading custom W or RB rule files.
    Accepts POST only.
    Handles both uploaded files and predefined files.
    """
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
    """
    Core processing route.
    Handles scenario decomposition, applies rules,
    manages user choices (apply rule, recap, exceptions, etc.).
    Accepts GET and POST.
    """
    scenario = (request.form.get('scenario') or request.args.get('scenario') or "").strip()             # Scenario in the system, already broken down
    resultat = request.form.get("resultat") or request.args.get("resultat") or ""             # Breakdown of this scenario
    scenario1 = (request.form.get("scenario1") or request.args.get("scenario1") or "").strip()           # New scenario if the user has provided one
    complement = request.form.get("complement") or request.args.get("complement")         # Supplement to the prompt if the user is not satisfied with the breakdown

    if scenario1 != "":
        default_w_path, default_rb_path, W_files, RB_files = get_files(session)
        shutil.copy(session.get("original_rb_path", default_rb_path), os.path.join(app.config["UPLOAD_FOLDER"], "RB_working.txt"))              # We reset the RB_working work file
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
                       "rb"])               # reset session
        scenario = scenario1
    if scenario == "":              # Empty scenario case
        session["resultat"] = get_log(1)
        return render_template(html_file)
        
    choice = request.form.get("user_choice", None)              # We retrieve the user's choice if there is one (choice of which rule to apply)
    Rb = init_RB(session)              # Initialization of the rule base

    if not session.get("decomposition") or (complement != None):                # If we have the decomposition of S in memory or if the decomposition is not satisfactory
        premises = ';'.join(Rb.Var_dictionnary._variables.keys())
        prompt = get_prompt(scenario,premises,session)

        if (complement != None):
            S_join = ';'.join(session.get("decomposition"))
            prompt += get_complement(S_join,complement,session)
        
        resultat = call_llm(prompt,MODELS,clients,session,first_api,noms)
    
        S = resultat.split(";")

        session["decomposition"] = S
        session["appliquees"] = []
        premier_log = True               # For the beginning of the log, as the translation does not work in app.py
        log = ""
    else :
        S = session.get("decomposition", [])
        log = request.form.get("log") or request.args.get("log") or ""
        premier_log = False

    deja_appliquees = session.get("appliquees", [])

    if (choice is not None) and (choice != "-1") and (choice != "-2") and (choice != "-3"):                 # If the user has chosen
        # Add the rule directly, then move on to the next step
        deja_appliquees = session.get("appliquees", [])

        choice = int(choice)
        ccl = Rb.rules[choice].conclusion
        for c in ccl:                   # Negations are translated into strings so that they can be entered into S
            if isinstance(c,Not):
                temp = "~"+c.children[0].name
            else:
                temp = c.name
            if temp not in S:
                S.append(temp)

        log += "\n\n"
        log += f" {get_log(2)} {choice} : {Rb.rules_original[choice]} "

        deja_appliquees.append(choice)
        session["appliquees"] = deja_appliquees

    analyse = scenario_check_web4_test(S, Rb,deja_appliquees,premier_log)                # Applicable rules
    indices = analyse.get("indices",[])              # Indexes of the rules
    options = analyse.get("options",[])
    output = analyse.get("output","")

    session["resultat"]=resultat
    session["scenario"]=scenario

    if len(indices)==0 and len(deja_appliquees) == 0:              # If no rule is applicable and no rule has been applied, error message
        return render_template(html_file,
                                No_rule = True,
                                log="No log")
    elif choice=="-1":              # exception generation
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
    elif choice=="-3":              # whether or not the base used has to be changed
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
    else:               # If rules are possible, choices are transmitted. 
        # (if no rule is possible, we can then request a recap, or look for an exception to the last rule applied)
        log +="\n".join(output)
        pas_premier = True
        no_choice = False
        if len(deja_appliquees)==0:
            pas_premier = False
        if options == []:
            no_choice = True
        return render_template(html_file,
                               conflit=True,
                               options=options,
                               no_choice = no_choice,
                               indices=indices,
                               log=log,
                               pas_premiere_regle=pas_premier)

#------------------------------------------------------------------------------------------------------------------------#

@app.route("/exceptions", methods=["POST"])
def exception():
    """
    Route to handle exceptions to rules.
    Computes possible adaptations and displays them to the user.
    Accepts POST only.
    """

    # ----------------------------
    # Step 1: Retrieve method choices
    # ----------------------------
    DISTANCE_METHODS = get_distance_method()                 # available distance metrics (dict: label -> function)
    SELECTION_METHODS = get_selection_method()              # available selection strategies (dict: label -> function)

    # Get chosen method labels from form (fallback = first available option)
    dist_choice_label = request.form.get("distances", list(DISTANCE_METHODS.keys())[0]).strip()
    sel_choice_label = request.form.get("selection", list(SELECTION_METHODS.keys())[0])

    # Map chosen labels to actual functions
    distance_method = DISTANCE_METHODS[dist_choice_label]
    selection_method = SELECTION_METHODS[sel_choice_label]

    # ----------------------------
    # Step 2: Gather user input and session state
    # ----------------------------
    scenario = request.form.get('scenario', "").strip()             # the current scenario being analyzed
    log = request.form.get("log", "")               # log/history of actions
    seuil = request.form.get("seuil")               # optional threshold parameter

    deja_appliquees = session["appliquees"]             # list of rules already applied

    # ----------------------------
    # Step 3: Build arguments for exception search
    # ----------------------------
    if seuil is not None:
        # if threshold is provided, include it as argument
        arguments = list([selection_method,seuil])
    else:
        # otherwise only pass the selection method
        arguments = list([selection_method])

    # ----------------------------
    # Step 4: Initialize Rule Base (RB) with current state S
    # ----------------------------
    resultat = request.form.get("resultat", "")
    S = resultat.split(";")                  # situation is represented as list of propositions
    Rb = init_RB(session)               # load current rule base
    Rb.init_S(S)                # initialize with state S

    # ----------------------------
    # Step 5: Compute possible exceptions
    # ----------------------------
    # Find exception/adaptation candidates for the last applied rule
    choix_ex = choix_exception(distance_method, Rb, arguments,deja_appliquees[-1])

    # If no adaptation found, return directly with message
    if choix_ex["options"] == []:
        log += choix_ex["output"]
        return render_template(
            html_file,
            Pas_adaptation = True,              # flag in template: no adaptation available
            log = log)

    # ----------------------------
    # Step 6: Build readable strings for adaptations
    # ----------------------------
    # We copy data to avoid mutating the original structures
    adaptations_string = copy.deepcopy(choix_ex["regles_adaptees"])
    adaptations_list = copy.deepcopy(choix_ex["regles_adaptees"])

    # Iterate through each adapted rule and its exceptions
    for i,regle in enumerate(choix_ex["regles_adaptees"]):
        for j,exception in enumerate(regle) :
            P,C = exception             # Premises and Conclusion of the adapted rule

            # Convert premises and conclusions into readable forms (lists + strings)
            P_list = [str(v) for v in P]
            C_list = [str(v) for v in C]
            P_str = ' ^ '.join(str(s) for s in P)               # join premises with AND operator
            C_str = ' ^ '.join(str(c) for c in C)               # same for conclusions

            # Update string representation for display in template
            adaptations_string[i][j] = (f"{P_str} => {C_str}")
            # Save structured lists for later reuse (stored in session)
            adaptations_list[i][j] = [P_list,C_list]
    session["adaptations"]=adaptations_list

    # ----------------------------
    # Step 7: Render template with results
    # ----------------------------
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
    """
    Route to handle user’s choice of proposed exception/adaptation.
    Updates rule base and redirects to processing page.
    Accepts POST only.
    """

    # ----------------------------
    # Step 1: Retrieve form/session data
    # ----------------------------
    scenario = request.form.get('scenario', "").strip()
    log = request.form.get("log", "")

    choix = request.form.get("choix_exception")             # user’s choice of exception (index pair "i|j" or -1)
    deja_appliquees = session["appliquees"]                 # rules already applied

    resultat = request.form.get("resultat", "")
    S = resultat.split(";")

    # ----------------------------
    # Step 2: Case: user cancelled (choix == -1)
    # ----------------------------
    if choix == "-1":
        log +=f"\n \n {get_log(3)} "                # Append cancellation log and redirect to traiter()
        return redirect(url_for(
                "traiter",
                scenario=scenario,
                log=log,
                resultat=resultat
            ))
    
    # ----------------------------
    # Step 3: Case: user selected an exception/adaptation
    # ----------------------------
    else:
        session["new_rules"] = True             # mark that new rules have been added (At the end, we will therefore propose creating a new Rule Base with these rules)
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
