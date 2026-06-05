from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.dtos import (
    ClauseDTO,
    RetrievalResultDTO
)


class RAGRetrievalAgent:

    def __init__(
        self,
        persist_directory: str = "vectorstore/",
        threshold: float = 0.5
    ):
        self.threshold = threshold

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def retrieve(
        self,
        clause: ClauseDTO,
        k: int = 5
    ) -> list[RetrievalResultDTO]:

        docs = self.vectorstore.similarity_search_with_score(
            clause.text,
            k=k
        )

        results = []

        for doc, score in docs:

            if score > 0.75:
                continue

            results.append(
                RetrievalResultDTO(
                    text=doc.page_content,
                    source=doc.metadata.get(
                        "source",
                        "unknown"
                    ),
                    score=float(score)
                )
            )

        return results