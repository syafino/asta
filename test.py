# Task Implementation: Neural Reranker for Scholarly Documents
from sentence_transformers import CrossEncoder

def scholarly_reranker(query, candidate_papers):
    #Reranks a list of candidate papers based on semantic relevance to the query 

    # Load a SOTA cross-encoder model
    model = CrossEncoder('BAAI/bge-reranker-v2-m3')
    
    # Format the input as pairs: [Query, Document]
    sentence_pairs = [[query, paper["title"]] for paper in candidate_papers]
    
    # Generate relevance scores via simultaneous self-attention
    scores = model.predict(sentence_pairs)
    
    # Attach scores to papers and sort descending
    for i, paper in enumerate(candidate_papers):
        paper["rerank_score"] = scores[i]
        
    ranked_results = sorted(candidate_papers, key=lambda x: x["rerank_score"], reverse=True)
    return ranked_results

if __name__ == "__main__":
    # The user's specific natural language query
    user_query = "What are the structural differences in amyloid plaque types in early-onset Alzheimer's?"
    
    # Mocking the output of Phase 1 (BM25/Vector Search) which is noisy
    retrieved_candidates = [
        {"id": 1, "title": "Alzheimer's - Looking beyond plaques (Griffin, 2011)"},
        {"id": 2, "title": "In vivo imaging reveals sigmoidal growth kinetic of beta-amyloid plaques (Burgold, 2014)"},
        {"id": 3, "title": "The coarse-grained plaque: a divergent A-beta plaque-type in early-onset Alzheimer’s disease (Boon, 2020)"},
        {"id": 4, "title": "The neuropathological diagnosis of Alzheimer’s disease (DeTure, 2019)"}
    ]
    
    # Run the reranker
    final_ranking = scholarly_reranker(user_query, retrieved_candidates)
    
    print(f"Query: '{user_query}'\n")
    print("--- Reranked Results ---")
    for rank, paper in enumerate(final_ranking, 1):
        print(f"Rank {rank} (Score: {paper['rerank_score']:.4f}) | {paper['title']}")
