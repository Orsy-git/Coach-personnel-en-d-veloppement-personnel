# mon_coach_llm/llm_rag.py

import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Charger les variables d'environnement (y compris votre clé OpenAI)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Vérifier si la clé API est présente
if not OPENAI_API_KEY:
    raise ValueError("La clé API OpenAI n'est pas définie. Assurez-vous d'avoir OPENAI_API_KEY dans votre fichier .env")

def get_rag_response(question: str) -> str:
    """
    Récupère la réponse du LLM augmentée par les documents pertinents du corpus.
    """
    # 2. Initialiser le modèle d'embeddings (le même que pour la création de la DB)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # 3. Charger la base de données vectorielle ChromaDB
    # Assurez-vous que 'persist_directory' correspond à l'endroit où votre DB a été sauvegardée.
    try:
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        print("Base de données vectorielle chargée avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement de la base de données ChromaDB : {e}")
        return "Désolé, je n'ai pas pu charger la base de données de connaissances."

    # 4. Initialiser le modèle de langage (LLM) d'OpenAI
    # 'gpt-3.5-turbo' est un bon point de départ. Vous pouvez essayer 'gpt-4' si vous avez accès.
    # 'temperature' contrôle la créativité de la réponse (0.0 = très factuel, 1.0 = très créatif)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=OPENAI_API_KEY)

    # 5. Créer la chaîne de Retrieval-Augmented Generation (RAG)
    # Le type de chaîne "stuff" prend tous les documents récupérés et les met dans le prompt.
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Récupère les 3 documents les plus pertinents
        return_source_documents=False # Pour l'instant, on ne retourne que la réponse, pas les sources
    )

    # 6. Exécuter la chaîne avec la question de l'utilisateur
    print(f"Question de l'utilisateur : {question}")
    try:
        # Langchain utilise maintenant .invoke() pour exécuter les chaînes
        response = qa_chain.invoke({"query": question})
        # La réponse est dans le champ 'result'
        return response['result']
    except Exception as e:
        print(f"Erreur lors de la génération de la réponse par le LLM : {e}")
        # Une erreur courante est un problème de quota ou de clé API invalide
        if "rate limit" in str(e).lower() or "invalid api key" in str(e).lower():
            return "Désolé, il y a un problème avec l'API OpenAI (limite de requêtes ou clé invalide). Veuillez réessayer plus tard ou vérifier votre clé."
        return "Désolé, une erreur est survenue lors de la génération de la réponse."


if __name__ == "__main__":
    # Ceci est un exemple d'utilisation pour tester la fonction directement
    print("Test de la fonction get_rag_response()...")
    test_question = "Quels sont les principes de Jim Rohn sur la discipline ?"
    answer = get_rag_response(test_question)
    print("\n--- Réponse du Coach ---")
    print(answer)

    print("\nDeuxième test...")
    test_question_2 = "Comment Tony Robbins aide-t-il à changer son état émotionnel ?"
    answer_2 = get_rag_response(test_question_2)
    print("\n--- Réponse du Coach ---")
    print(answer_2)