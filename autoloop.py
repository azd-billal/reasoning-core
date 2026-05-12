import ollama
import re
import yaml # Ajout de la librairie YAML

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG_FILE = 'config.yaml' # Le chemin vers ton fichier de config
MAX_ITERATIONS = 3

def load_prompts_from_config(filepath):
    """Lit le fichier YAML et extrait les infos des modèles."""
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    coder_model = None
    reviewer_model = None
    prompt_coder = ""
    prompt_reviewer = ""
    base_url = config.get('ollamaBaseUrl', 'http://localhost:11434')

    for model_info in config.get('models', []):
        if model_info['name'] == 'Qwen Coder':
            coder_model = model_info['model']
            prompt_coder = model_info['systemMessage']
        elif model_info['name'] == 'DeepSeek Reviewer':
            reviewer_model = model_info['model']
            prompt_reviewer = model_info['systemMessage']

    return base_url, coder_model, reviewer_model, prompt_coder, prompt_reviewer

# Chargement dynamique au démarrage du script !
OLLAMA_BASE_URL, CODER_MODEL, REVIEWER_MODEL, PROMPT_CODER, PROMPT_REVIEWER = load_prompts_from_config(CONFIG_FILE)

# ==========================================
# UTILITAIRES
# ==========================================
def extract_code(text):
    matches = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
    return matches[0].strip() if matches else text.strip()

def clean_deepseek_r1_output(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

# ==========================================
# ORCHESTRATION MAIN LOOP
# ==========================================
def orchestrate_task(task_description, target_file):
    print(f"\n🚀 Lancement de la tâche : {task_description}")
    current_context = f"Tâche : {task_description}"

    client = ollama.Client(host=OLLAMA_BASE_URL)

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Itération {i}/{MAX_ITERATIONS} ---")

        # 1. QWEN
        print(f"🧠 [{CODER_MODEL}] Écriture du code en cours...")
        response_coder = client.chat(model=CODER_MODEL, messages=[
            {'role': 'system', 'content': PROMPT_CODER},
            {'role': 'user', 'content': current_context}
        ])
        clean_code = extract_code(response_coder['message']['content'])

        # 2. DEEPSEEK
        print(f"🔎 [{REVIEWER_MODEL}] Analyse de fiabilité en cours...")
        response_reviewer = client.chat(model=REVIEWER_MODEL, messages=[
            {'role': 'system', 'content': PROMPT_REVIEWER},
            {'role': 'user', 'content': f"Code à analyser :\n\n```\n{clean_code}\n```"}
        ])
        review_result = clean_deepseek_r1_output(response_reviewer['message']['content'])

        # 3. DÉCISION
        if "STATUS: PASS" in review_result:
            print(f"✅ [{REVIEWER_MODEL}] Code validé (STATUS: PASS) !")
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(clean_code)
            print(f"💾 Fichier sauvegardé : {target_file}")
            return True
        else:
            print(f"❌ [{REVIEWER_MODEL}] Faille détectée. Renvoi de la correction...")
            current_context = f"Tâche initiale : {task_description}\n\nVoici ton code précédent :\n```\n{clean_code}\n```\n\nLe Reviewer a rejeté ce code. Applique IMMÉDIATEMENT ce patch et renvoie le code :\n{review_result}"

    print(f"⚠️ Échec après {MAX_ITERATIONS} itérations.")
    return False

if __name__ == "__main__":
    tache = "Écris une fonction Python pour lire un fichier JSON..."
    fichier_sortie = "config_parser.py"
    orchestrate_task(tache, fichier_sortie)