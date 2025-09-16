# A Tool for Handling Unexpected Exceptions in Legal Reasoning

We have developed a system, based on prioritized default logic, for representing legal rules and their exceptions. Our system can help the law practitioner's reasoning in presence of scenarios that induce an exception to a rule. When there is no known exception to the rule *r1* for a scenario, the system can help the user to adapt the exception *r2'* to a general rule *r2*.

## Dependencies
The following Python libraries must be installed:
- `numpy`,
- `os`,
- `flask`,
- `mitralai`,
- `huggingface_hub`
- `werkzeug.utils`,
- `shutil`,
- `scipy.spatial`,
- `copy`.

You need to use at least one of those LLM APIs:
- Mistral: create an account for the Mistral API on the section *La plateforme* of the [https://mistral.ai/fr/products/la-plateforme](Mistral website), then generate an API key,
- Hugging Face: create an account on [https://huggingface.co](Hugging Face website), and generate a personal access token.

To use your API keys, create a file named `cles_API.txt` in the project folder. Write the name of the API, followed by a colon and then the key as shown in the example bellow. Use the same names as in the example (names are case sensitive!):
```
Mistral : Mistral_key
Ollama : HuggingFace_key
```

You can choose to use only one of the APIs, or both (and let the user choose which one to call through the web interface).

## Running the project

To launch the project, simply open a terminal in the project folder and run the following command:
```
python app.py
```

## Web Interface Usage

Details on how to use the web interface are available in the [User_guide.pdf](User Guide).