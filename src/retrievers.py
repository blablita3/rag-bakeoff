from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize


# --- Hand Made Retrievers ---

class BaselineRetriever:
    """Trivial retriever to mantain logic""" 

    def invoke(self, query):
        return []


class HybridRRFRetriever:
    """
    Hybrid retriever. Combines a sparse retriever and a dense one, using reciprocal rank fusion.
    """

    def __init__(self, sparse_retriever, dense_retriever, top_k, k_rrf=60):

        self.sparse = sparse_retriever
        self.dense = dense_retriever

        # Docs to return
        self.top_k = top_k 
        
        # RFF constant
        self.k_rrf = k_rrf

    def rrf_merge(self, sparse_results, dense_results):
        """Perform Reciprocal Rank Fusion"""

        scores = {}
        all_docs = []

        # Assign RRF scores
        for results in [sparse_results, dense_results]:
            for rank, doc in enumerate(results):
                id = doc.metadata["id"]
                scores[id] = scores.get(id, 0) + 1 / (self.k_rrf + rank)
                all_docs.append(doc)

        # Order by RRF score and remove duplicates
        seen = set()
        final_docs = []
        for doc in sorted(all_docs, key=lambda d: scores[d.metadata["id"]], reverse=True):
            id = doc.metadata["id"]
            if id not in seen:
                final_docs.append(doc)
                seen.add(id)
            if len(final_docs) >= self.top_k:
                break
        
        return final_docs

    def invoke(self, query):
        """Retrieve top-k docs using sparse + dense retrievers with RRF fusion"""

        sparse_docs = self.sparse.invoke(query, k=self.top_k)
        dense_docs = self.dense.invoke(query, k=self.top_k)
        return self.rrf_merge(sparse_docs, dense_docs)


# --- Retriever Initialization ---

def initialize_retriever(chunks, vectorstore, retrieval_method, k):
    """
    Returns a retriever object for the given retrieval method (str).
    Needs chunks (list of Documents), a vectorstore (Croma object) and 
    a k param (int: number of retrieved docs) for initializating it.
    """
    
    # Separate cases
    if retrieval_method == "baseline":
        retriever = BaselineRetriever()

    if retrieval_method in ["bm25", "hybrid"]:
        
        # Initialize BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
        chunks,
        k=k,
        preprocess_fun=word_tokenize
        )

        if retrieval_method == "bm25":
            retriever = bm25_retriever
    

    if retrieval_method in ["dense", "hybrid"]:
        
        # Initialize dense retriever
        dense_retriever = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": k, "lambda_mult": 0.8}
        )
        
        if retrieval_method == "dense":
            retriever = dense_retriever
    
    # Initialize hybrid retriever
    if retrieval_method == "hybrid":
        retriever = HybridRRFRetriever(bm25_retriever, dense_retriever, k)

    # Add retrieval and chunking method info
    chunking_method = chunks[0].metadata["method"]
    retriever._chunking_method = chunking_method
    retriever._retrieval_method = retrieval_method

    return retriever
