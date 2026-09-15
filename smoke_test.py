from evals.rag_client import run_rag_pipeline

answer, context = run_rag_pipeline('What is the primary endpoint of the study?')
print('ANSWER:', answer[:300])
print('CONTEXT CHUNKS:', len(context))
