# scratch script, not a real test file
import asyncio
from rag_service.services.generation_service import RagGenerationService


async def main():
    svc = RagGenerationService.__new__(
        RagGenerationService
    )  # skip __init__, don't need retrieval_service for this
    result = await svc._check_ambiguity(
        query_text="What about corticosteroids?",
        conversation_history=[],
        formatted_context="[Protocol Ulcerative Colitis|p37|section:4.3.2 Prohibited Therapy]\n...",
    )
    print(result)


asyncio.run(main())
