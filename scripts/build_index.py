import os
import sys

from dotenv import load_dotenv
load_dotenv()


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tools.pdf_tools import load_corpus
from src.tools.vector_tools import build_index

if __name__ == "__main__":

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    corpus_path = os.path.join(base_dir, "corpus")
    docs = load_corpus(corpus_path)

    build_index(docs)
    vectorstore_dir = "vectorstore/"

    if os.path.exists(vectorstore_dir) and os.listdir(vectorstore_dir):
        print(f"Folderul '{vectorstore_dir}' exista deja si nu e gol")
        sys.exit(1)

    docs = load_corpus("corpus/")
    build_index(docs)