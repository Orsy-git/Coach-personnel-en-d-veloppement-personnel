# mon_coach_llm/llm_rag.py

import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    raise ValueError("La clé API OpenAI n'est pas définie.")

def get_rag_response(question: str) -> str:
    """
    Récupère la réponse du LLM augmentée par les documents pertinents du corpus.
    """
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

  
    try:
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        print("Base de données vectorielle chargée avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement de la base de données ChromaDB : {e}")
        return "Désolé, je n'ai pas pu charger la base de données de connaissances."

    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, openai_api_key=OPENAI_API_KEY)

    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), 
        return_source_documents=False 
    )

    
    print(f"Question de l'utilisateur : {question}")
    try:
        
        response = qa_chain.invoke({"query": question})
        
        return response['result']
    except Exception as e:
        print(f"Erreur lors de la génération de la réponse par le LLM : {e}")
       
        if "rate limit" in str(e).lower() or "invalid api key" in str(e).lower():
            return "Désolé, il y a un problème avec l'API OpenAI (limite de requêtes ou clé invalide). Veuillez réessayer plus tard ou vérifier votre clé."
        return "Désolé, une erreur est survenue lors de la génération de la réponse."


if __name__ == "__main__":
    
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