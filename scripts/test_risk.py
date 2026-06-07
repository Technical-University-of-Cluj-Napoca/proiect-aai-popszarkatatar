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

from src.agents.risk_agent import (
    RiskAssessmentAgent
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

    retrieval_agent = RAGRetrievalAgent()
    risk_agent = RiskAssessmentAgent()

    context = retrieval_agent.retrieve(clause)

    result = risk_agent.assess(
        clause,
        context
    )

    print(result)


if __name__ == "__main__":
    main()