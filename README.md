# Search & Recommendation System
# Industry Solution

![Untitled](Search%20&%20Recommendation%20System%208674791d93d24e30baa4d9177f4f7e1f/Untitled.png)

- **Recall:** retrieve **a few thousand** relevant items from the whole item pool.
- **Pre-Ranking**: get around **a few hundred** items. A lightweight model.
- **Ranking**: ****sophisticated ranking models to get around the top 50 items.
- **Reranking**: for extra business flexibility and performance.

## Recall

`recall`, also known as sensitivity, measures the proportion of actual positive cases that were correctly identified by the model To ensure that **the search query captures as many relevant documents as possible, even at the expense of retrieving some non-relevant ones**

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

In simple terms, it answers the question: "Out of all the relevant items, how many did the system successfully find?”

`precision` answers"Out of all the items the system identified as relevant, how many were actually relevant?”

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

## Pre-ranking 粗排

Pre-ranking, or coarse ranking, narrows down the candidate set from potentially millions of items to a smaller subset (e.g., hundreds or thousands) that are more likely to be relevant. This step balances between efficiency and effectiveness, ensuring that the computational heavy lifting in the next steps is manageable.

- **Dense Retrieval**: This approach uses deep learning models to encode both queries and documents into dense vectors in the same embedding space. The relevance of a document to a query is then estimated by the similarity between their vectors, often calculated with dot product or cosine similarity.
- **Two-stage Retrieval Systems**: These systems initially retrieve a broad set of documents using a simpler, faster model (like BM25 or a basic embedding model) and then apply a more complex model to refine this set. The first stage emphasizes recall, while the second stage focuses on precision.

## Ranking 精排

The ranking stage involves evaluating the relevance of each item in the pre-ranked subset to the user's query with more sophisticated algorithms, often employing machine learning models. This stage aims to order the items by relevance as accurately as possible.

- **Multiple Negative Ranking Loss**: This is a training strategy used in machine learning models for ranking. The model is trained on positive pairs (a query and its correct response) and several negative pairs (the same query paired with incorrect responses). The loss function is designed to increase the model's output for positive pairs and decrease it for negative pairs, effectively teaching the model to distinguish between relevant and irrelevant items more accurately.

## Re-ranking 重排

Re-ranking is the final adjustment of the results before presenting them to the user. It might take into account additional context, user preferences, diversity requirements, or freshness of the content. This stage fine-tunes the list of items to ensure the top results are the most relevant and appealing to the user.

- **Relevance Scoring**: In re-ranking, each item-query pair is assigned a relevance score, which may incorporate a wide range of signals, including user engagement metrics, semantic relevance, and personalization factors. Advanced machine learning models, such as gradient-boosted decision trees or deep neural networks, can be used for scoring.

## Benchmark

### NDCG (Normalized Discounted Cumulative Gain)

**Discounted cumulative gain** (**DCG**) is a measure of ranking quality. See [wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)

$$
\text{DCG}@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

where $rel_i$ is the relevance score of the document at position i (with the top position being i=1)

$$
\text{NDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k} = \frac{\sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}}{\sum_{i=1}^{k} \frac{2^{rel_{(i)}} - 1}{\log_2(i+1)}}
$$

where $rel_{(i)}$ is relevance score of the ith document in the ideal ordering. 

![Untitled](Search%20&%20Recommendation%20System%208674791d93d24e30baa4d9177f4f7e1f/Untitled%201.png)

### MAP (Mean Average Precision)

MAP is a measure of precision averaged across all relevant documents and queries, accounting for the order of the documents, see [tutorial](https://www.evidentlyai.com/ranking-metrics/mean-average-precision-map).

For a single query, Average Precision (AP) is calculated as:

$$
\text{AP} = \frac{\sum_{k=1}^{n} (\text{Precision}(k) \times rel(k))}{N}
$$

- P(k) is the precision at cut-off k in the list
- rel(k) is an indicator function equaling 1 if the item at rank k is relevant, otherwise 0
- n is the number of retrieved documents.

For multiple queries, we calculate mean of AP.

### Recall@K

See [tutorial](https://www.notion.so/Search-Recommendation-System-8674791d93d24e30baa4d9177f4f7e1f?pvs=21). 

$$
\text{Recall}@k = \frac{\text{number of relevant documents retrieved in top } k}{\text{total number of relevant documents}}
$$

# Research

## **1. RankGPT**

Paper :[<**Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents**>](https://arxiv.org/abs/2304.09542)

**Training**

1. randomly sample 10K queries from the `MS MARCO` training set, and each query is retrieved by BM25 with 20 candidate passages.
2. three types of instructions
    1. `query generation` 
    2. `relevance generation` 
    3. `permutaiton generation`
        
        ![Untitled](Search%20&%20Recommendation%20System%208674791d93d24e30baa4d9177f4f7e1f/Untitled%202.png)
        
3. To handle context length problem, use `sliding window` 
    
    ![Untitled](Search%20&%20Recommendation%20System%208674791d93d24e30baa4d9177f4f7e1f/Untitled%203.png)
    
4. Training objectives
- `RankNet` is a pairwise loss that measures the cor- rectness of relative passage orders. We can construct M (M − 1)/2 pairs.

$$
\mathcal{L}_{\text{RankNet}} = \sum_{i=1}^{M} \sum_{j=1}^{M} \mathbf{1}_{r_i < r_j} \log(1 + \exp(s_i - s_j))
$$

**Benchmark**

- TREC : (i) TREC-DL19 contains 43 queries, (ii) TREC-DL20 contains 54 queries
- BEIR: Covid, NFCorpus, Touche, DBPedia, SciFact, Signal, News, Robust04
- Mr.TyDi: uses the first 100 samples in the test set of each language
    - **topics**: Queries used to evaluate IR system performance. Ex: `19335: {'title': 'anthropological definition of environment'}`
    - **Qrels**: short for "query relevance judgments". Ex: `{19335: {1017759: '0', 1082489: '0'}}`
        - 19335 is the query id
        - 1017759 and 1082489 are the document IDs
        - '0'-'3': relevance score, 0 means no relevance

## **2. monoT5**

Paper :[**<Document Ranking with a Pretrained Sequence-to-Sequence Model>**](https://arxiv.org/abs/2003.06713)

## 3. Google RankT5

Paper: [<**RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses**>](https://arxiv.org/abs/2210.10634)

## 4. **TSRankLLM**

Paper :<**[TSRankLLM: A Two-Stage Adaptation of LLMs for Text Ranking](https://arxiv.org/abs/2311.16720)>**

1. continual pre-training (CPT) on LLMs by using a large-scale weakly-supervised corpus.
2. SFT with RankLLaMA

![Untitled](Search%20&%20Recommendation%20System%208674791d93d24e30baa4d9177f4f7e1f/Untitled%204.png)

## 5. **AgentSearch**

[API Doc](https://agent-search.readthedocs.io/en/latest/api/main.html)

# Reference

- https://github.com/frutik/awesome-search

[https://github.com/snexus/llm-search](https://github.com/snexus/llm-search)

- https://github.com/snexus/llm-search
- [Retrieval Augmented Generation with Huggingface Transformers and Ray](https://huggingface.co/blog/ray-rag)
- [https://www.pinecone.io/learn/series/rag/rerankers/](https://www.pinecone.io/learn/series/rag/rerankers/)
- [https://github.com/ielab/llm-rankers](https://github.com/ielab/llm-rankers) [https://www.sbert.net/examples/applications/retrieve_rerank/README.html#re-ranker-cross-encoder](https://www.sbert.net/examples/applications/retrieve_rerank/README.html#re-ranker-cross-encoder)
- [https://github.com/leptonai/search_with_lepton](https://github.com/leptonai/search_with_lepton)
- [https://txt.cohere.com/using-llms-for-search/](https://txt.cohere.com/using-llms-for-search/)
- [https://developers.google.com/machine-learning/recommendation](https://developers.google.com/machine-learning/recommendation)
- https://github.com/SciPhi-AI/agent-search