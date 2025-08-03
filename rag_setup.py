# mon_coach_llm/rag_setup.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    raise ValueError("La clé API OpenAI n'est pas définie. Assurez-vous d'avoir OPENAI_API_KEY dans votre fichier .env")

def create_vector_db(data_path="corpus/"):
    """
    Charge les documents, les découpe, génère les embeddings et les stocke dans ChromaDB.
    """
    documents = []
    print(f"Chargement des documents depuis le dossier : {data_path}")
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_path, filename)
            try:
                
                loader = TextLoader(filepath, encoding="utf-8")
                documents.extend(loader.load())
                print(f"  - Chargé : {filename}")
            except Exception as e:
                print(f"  - Erreur lors du chargement de {filename} : {e}")

    if not documents:
        print("Aucun document trouvé ou chargé. Assurez-vous que le dossier 'corpus' contient des fichiers .txt valides.")
        return

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     
        chunk_overlap=200    
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents divisés en {len(chunks)} morceaux.")

    
    print("Génération des embeddings avec OpenAI...")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    
    print("Stockage des embeddings dans ChromaDB (./chroma_db)...")
    vector_db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )
    vector_db.persist()
    print("Base de données vectorielle créée/mise à jour avec succès dans ./chroma_db")

if __name__ == "__main__":
    create_vector_db()