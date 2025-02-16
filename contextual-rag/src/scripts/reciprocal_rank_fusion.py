class ReciprocalRankFusion:
    def __init__(self, semantic_weight: float = 0.6, bm25_weight: float = 0.4):
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def reciprocal_rank_fusion(self, chunk_ids, ranked_chunk_ids, ranked_bm25_chunk_ids):
        """
        Fuse rank from multiple IR systems using Reciprocal Rank Fusion.
        Reciprocal Rank Fusion to merge the results from the BM25 search with the semantic search results. 
        This allows us to perform a hybrid search across both our BM25 corpus and vector DB to return the most optimal (documents) chunk for a given query.

        Formula:
            RRF(d) = Σ(r ∈ R) 1 / (k + r(d))
            d: a document but in this case a chunk
            k: constant (default 60)
            R: the set of rankers (retrievers)
            r(d): the rank / index of (document) chunk in ranker r

        Inputs:
            chunk_ids: list of combined chunk ids from BM25 and vector embedding
            ranked_chunk_ids: retrieved chunk ids from vector embedding
            ranked_bm25_chunk_ids: retrieved chunk ids from BM25

        Output:
            A dict consisiting of the weighted scoring for each chunk id
        """
        chunk_id_to_score = {}
        k = 60   # default K=60
        for chunk_id in chunk_ids:
            score = 0
            if chunk_id in ranked_chunk_ids:
                index = ranked_chunk_ids.index(chunk_id)                # index = rank
                score += self.semantic_weight * (1 / (index + k))       # Weighted scoring for vector embedding (semantic)   
            if chunk_id in ranked_bm25_chunk_ids:
                index = ranked_bm25_chunk_ids.index(chunk_id)
                score += self.bm25_weight * (1 / (index + k))           # Weighted scoring for BM25 (keyword)
            chunk_id_to_score[chunk_id] = score
        return chunk_id_to_score