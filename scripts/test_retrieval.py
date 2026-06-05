import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.dtos import (
    ClauseDTO,
    ClauseType
)

from src.agents.retrieval_agent import (
    RAGRetrievalAgent
)


def main():

    clause = ClauseDTO(
        id="1",
        section="GDPR",
        text="""
        Operatorul poate prelucra datele personale
        ale clientului pe durata contractului.
        """,
        page=1,
        type=ClauseType.DATE_PERSONALE
    )

    agent = RAGRetrievalAgent()
    print("Vectorstore loaded")
    print(agent.vectorstore._collection.count())
    results = agent.retrieve(clause)

    print("Numar rezultate:", len(results))
    print(results)

    for result in results:
        print("=" * 50)
        print("Source:", result.source)
        print("Score:", result.score)
        print(result.text[:300])


if __name__ == "__main__":
    main()