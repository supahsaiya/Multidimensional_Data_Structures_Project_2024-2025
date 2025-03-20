
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
import numpy as np

def vectorize_reviews(processed_reviews):
    """
    Convert processed reviews into TF-IDF vectors.
    :param processed_reviews: List of processed reviews (lists of tokens).
    :return: TF-IDF matrix and feature names.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_reviews)
    return tfidf_matrix, vectorizer.get_feature_names_out()

def build_lsh(tfidf_matrix, num_perm=128):
    """
    Build an LSH index using MinHash for the given TF-IDF matrix.
    :param tfidf_matrix: TF-IDF matrix of reviews.
    :param num_perm: Number of permutations for MinHash.
    :return: LSH index and list of MinHash objects.
    """
    lsh = MinHashLSH(threshold=0.3, num_perm=num_perm)
    minhashes = []

    for i in range(tfidf_matrix.shape[0]):
        m = MinHash(num_perm=num_perm)
        for idx in tfidf_matrix[i].nonzero()[1]:
            m.update(str(idx).encode('utf8'))
        lsh.insert(f"doc_{i}", m)
        minhashes.append(m)

    return lsh, minhashes

def query_lsh(lsh, minhashes, query_index, top_n=5):
    """
    Query the LSH index for similar reviews.
    :param lsh: LSH index.
    :param minhashes: List of MinHash objects.
    :param query_index: Index of the query document.
    :param top_n: Number of similar documents to return.
    :return: List of similar document indices.
    """
    query_minhash = minhashes[query_index]
    similar_docs = lsh.query(query_minhash)
    return similar_docs[:top_n]

def test_lsh_pipeline(data):
    """
    Test the LSH pipeline with the given dataset.
    :param data: DataFrame with processed reviews.
    """
    # Vectorize reviews
    print("Vectorizing reviews...")
    tfidf_matrix, feature_names = vectorize_reviews(data['review'])

    # Build LSH index
    print("Building LSH index...")
    lsh, minhashes = build_lsh(tfidf_matrix)

    # Query for similar reviews
    print("Querying LSH for similar reviews...")
    query_index = 0  # Query the first document
    similar_docs = query_lsh(lsh, minhashes, query_index)
    print("Similar Reviews Indices:", similar_docs) # Debugging

    print(f"Reviews similar to document {query_index}:")
    for doc_id in similar_docs:
         # Strip "doc_" prefix and convert to integer
        doc_num = int(doc_id.replace("doc_", ""))
        print(f"Document {doc_num}: {data['review'].iloc[doc_num]}")


