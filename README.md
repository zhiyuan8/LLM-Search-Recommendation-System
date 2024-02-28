# Search

![Untitled](LLM%20for%20Search%20&%20Recommendation%208674791d93d24e30baa4d9177f4f7e1f/Untitled%201.png)

## Overrall

- **Recall:** This is the first stage of the recommendation flow. We usually retrieve **a few thousand** relevant items from the whole item pool.
- **Pre-Ranking**: We will get around a few hundred items using pre-ranking models. The pre-rank model is usually a lightweight model that imitates the ranking model.
- **Ranking**: ****In this stage, we use sophisticated ranking models to get around the top 50 items and show them to users.
- **Reranking**: We can add a reranking stage for extra business flexibility and performance. Reranking models can be very complex since they only need to rank a few top items.

## Recall

`recall`, also known as sensitivity, measures the proportion of actual positive cases that were correctly identified by the model. In the context of a search engine, it would measure the fraction of relevant documents that are retrieved out of all relevant documents that exist in the data set. 

The rationale for optimizing recall first is to ensure that **the search query captures as many relevant documents as possible, even at the expense of retrieving some non-relevant ones**

The formula for the recall is:

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

In simple terms, it answers the question: "Out of all the relevant items, how many did the system successfully find?”

In contrast, `precision` answers"Out of all the items the system identified as relevant, how many were actually relevant?”

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

## Pre-ranking

Pre-ranking, or coarse ranking, narrows down the candidate set from potentially millions of items to a smaller subset (e.g., hundreds or thousands) that are more likely to be relevant. This step balances between efficiency and effectiveness, ensuring that the computational heavy lifting in the next steps is manageable.

- **Dense Retrieval**: This approach uses deep learning models to encode both queries and documents into dense vectors in the same embedding space. The relevance of a document to a query is then estimated by the similarity between their vectors, often calculated with dot product or cosine similarity.
- **Two-stage Retrieval Systems**: These systems initially retrieve a broad set of documents using a simpler, faster model (like BM25 or a basic embedding model) and then apply a more complex model to refine this set. The first stage emphasizes recall, while the second stage focuses on precision.

## Ranking

The ranking stage involves evaluating the relevance of each item in the pre-ranked subset to the user's query with more sophisticated algorithms, often employing machine learning models. This stage aims to order the items by relevance as accurately as possible.

- **Multiple Negative Ranking Loss**: This is a training strategy used in machine learning models for ranking. The model is trained on positive pairs (a query and its correct response) and several negative pairs (the same query paired with incorrect responses). The loss function is designed to increase the model's output for positive pairs and decrease it for negative pairs, effectively teaching the model to distinguish between relevant and irrelevant items more accurately.

## Re-ranking

Re-ranking is the final adjustment of the results before presenting them to the user. It might take into account additional context, user preferences, diversity requirements, or freshness of the content. This stage fine-tunes the list of items to ensure the top results are the most relevant and appealing to the user.

- **Relevance Scoring**: In re-ranking, each item-query pair is assigned a relevance score, which may incorporate a wide range of signals, including user engagement metrics, semantic relevance, and personalization factors. Advanced machine learning models, such as gradient-boosted decision trees or deep neural networks, can be used for scoring.

## **RankGPT**

Permutation generation and sliding window for reranking

![Untitled](LLM%20for%20Search%20&%20Recommendation%208674791d93d24e30baa4d9177f4f7e1f/Untitled%202.png)

**Sliding Window**

Window size (w) and step size (s). We first use the LLMs to rank the passages from the (M − w)-th to the M-th. Then, we slide the win- dow in steps of s and re-rank the passages within the range from the (M − w − s)-th to the (M − s)- th.

![Untitled](LLM%20for%20Search%20&%20Recommendation%208674791d93d24e30baa4d9177f4f7e1f/Untitled%203.png)

## Cohere Rerank LLM

![Untitled](LLM%20for%20Search%20&%20Recommendation%208674791d93d24e30baa4d9177f4f7e1f/Untitled%204.png)

```jsx
results = co.rerank(query=item["query"], documents=documents, top_n=3, model="rerank-multilingual-v2.0")
```

Here is [https://cohere.com/pricing](https://cohere.com/pricing)

| Model | Cost |
| --- | --- |
| Default | $1.00 /1K SEARCHES |
| Fine-tuned | $1.00 /1K SEARCHES |

# Reference

## Search in general
- [Github awesome search]https://github.com/frutik/awesome-search
- [https://github.com/snexus/llm-search](https://github.com/snexus/llm-search)
- https://github.com/snexus/llm-search
- [Meituan (Mandarin)](https://tech.meituan.com/2022/08/11/coarse-ranking-exploration-practice.html)
- [Recommendation system(mandarin)](https://blog.csdn.net/qq_41750911/article/details/124573064)

## Search - RAG
-[Retrieval Augmented Generation with Huggingface Transformers and Ray](https://huggingface.co/blog/ray-rag)
-[LLM with semantic search](https://github.com/ArslanKAS/Large-Language-Models-with-Semantic-Search)

## Search - retrieval & rerank
-[Retrieve & Re-Rank — Sentence-Transformers  documentation](https://www.sbert.net/examples/applications/retrieve_rerank/README.html#re-ranker-cross-encoder)
-[Pinecone rerankers](https://www.pinecone.io/learn/series/rag/rerankers/)
-[llm-rankers](https://github.com/ielab/llm-rankers)
-[RankGPT: LLMs as Re-Ranking Agent](https://github.com/sunnweiwei/RankGPT)

## Search - LLM examples
- [https://github.com/leptonai/search_with_lepton](https://github.com/leptonai/search_with_lepton)
- [https://txt.cohere.com/using-llms-for-search/](https://txt.cohere.com/using-llms-for-search/)
- [https://github.com/SciPhi-AI/agent-search](https://github.com/SciPhi-AI/agent-search)