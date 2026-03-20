# Task Implementation: Neural Reranker for Scholarly Documents
from sentence_transformers import CrossEncoder
import bibtexparser

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

def load_papers_from_bib(bib_file):
    # Parses a .bib file and returns a list of candidate paper dicts.
    with open(bib_file, "r", encoding="utf-8") as f:
        bib_database = bibtexparser.load(f)

    candidates = []
    seen_titles = set()
    for i, entry in enumerate(bib_database.entries):
        title = entry.get("title", "Untitled")
        if title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        author = entry.get("author", "Unknown")
        year = entry.get("year", "N/A")
        candidates.append({
            "id": len(candidates) + 1,
            "title": f"{title} ({author.split(' and ')[0]}, {year})"
        })
    return candidates

if __name__ == "__main__":
    # The user's specific natural language query
    user_query = "What are the structural differences in amyloid plaque types in early-onset Alzheimer's?"

    # Load candidates from papers.bib instead of hardcoding
    retrieved_candidates = load_papers_from_bib("papers.bib")

    # Run the reranker
    final_ranking = scholarly_reranker(user_query, retrieved_candidates)

    print(f"Query: '{user_query}'\n")
    print("--- Reranked Results ---")
    for rank, paper in enumerate(final_ranking, 1):
        print(f"Rank {rank} (Score: {paper['rerank_score']:.4f}) | {paper['title']}")
