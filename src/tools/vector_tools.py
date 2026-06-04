from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def build_index(documents):

    text_splitter = RecursiveCharacterTextSplitter (
        chunk_size=300,
        chunk_overlap=50
    )
    #Am ales chunk_size de 300 de caractere pentru a izola mai bine clauzele juridice individuale.
    #Daca foloseam o valoare mai mare riscam ca un chunk sa contina mai multe articole de lege iar LLM-ul sa se piarda in detlii
    #Overlap-ul de 50 asigura ca frazele care se intind pe doua chunk-uri sa nu isi piarda contextul
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore/"
    )

    return vectorstore