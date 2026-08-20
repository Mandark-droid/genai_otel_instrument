# Retrieval quality

TraceVerde emits the existing `retrieval.*` and `db.vector.*` attributes for
the retrieved documents and requested top-k. The additive `rag.*` attributes
describe quality and provenance signals that an application or vector
instrumentor can know at retrieval time.

## Quality attributes

| Attribute | Meaning |
|---|---|
| `rag.embedding.model` | Model used for the query embedding |
| `rag.embedding.index_model` | Model used to build the index |
| `rag.embedding.model_match` | Boolean comparison of the two model identifiers |
| `rag.embedding.dim` | Query/index vector dimension |
| `rag.search.score_floor` | Minimum score accepted by the search |
| `rag.search.distance` | Distance or similarity metric name |
| `rag.result.score_max` / `min` / `mean` | Distribution summary for returned scores |
| `rag.result.score_margin` | Difference between the two highest scores |
| `rag.corpus.version` | Corpus or index version |
| `rag.context.tokens_est` | Estimated context tokens sent downstream |
| `rag.context.truncated` | Whether context was truncated |
| `rag.answer.refused` | Whether the answer path refused to answer |

`top_k` and result count remain `db.vector.top_k` and
`retrieval.document_count`; TraceVerde does not emit duplicate `rag.*` names
for those values.

## Application-owned signals

Use the public helper when the application owns the embedding, index metadata,
or answer policy:

```python
instrumentor.add_retrieval_quality_attributes(
    span,
    embedding_model="text-embedding-3-small",
    index_embedding_model="text-embedding-3-small",
    embedding_dim=1536,
    distance="cosine",
    scores=[0.91, 0.84, 0.72],
    corpus_version="2026-08-20",
    context_tokens_est=1200,
    context_truncated=False,
    answer_refused=False,
)
```

`gen_ai.rag.context` is not emitted by the library. It is an application-set
input consumed by the evaluation processors, so applications that use
hallucination evaluation should continue to set it explicitly.
