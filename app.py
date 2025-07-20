from flask import Flask, request, render_template
from llm_rag import get_rag_response # Importez la fonction RAG que nous venons de tester
import os
from dotenv import load_dotenv

# Charger les variables d'environnement (y compris OPENAI_API_KEY)
load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    response_text = None
    user_question = ""
    if request.method == "POST":
        user_question = request.form["question"]
        if user_question:
            print(f"Requête reçue de l'utilisateur via Flask : {user_question}")
            response_text = get_rag_response(user_question)
            print(f"Réponse générée par le coach : {response_text[:100]}...") # Affiche un extrait

    # Passer la question de l'utilisateur et la réponse au template HTML
    return render_template("index.html", response=response_text, user_question=user_question)

if __name__ == "__main__":
    # Cette ligne s'assure que la base de données vectorielle est déjà là.
    # N'exécutez 'rag_setup.py' que si vous voulez recréer/mettre à jour la DB.
    # Vous l'avez déjà fait, donc pas besoin de l'appeler ici.
    app.run(debug=True)