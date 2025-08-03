from flask import Flask, request, render_template
from llm_rag import get_rag_response 
import os
from dotenv import load_dotenv

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
            print(f"Réponse générée par le coach : {response_text[:100]}...") 

    
    return render_template("index.html", response=response_text, user_question=user_question)

if __name__ == "__main__":
    app.run(debug=True)