# AI6130 Group Project Report Guide

## Method

### Datasets

HotpotQA, PubMedQA, and DelusionQA from RAGBench [[Dataset](https://huggingface.co/datasets/galileo-ai/ragbench), [GitHub](https://github.com/YourUsername/RAGBench), [Paper](https://arxiv.org/pdf/2407.11005v2)]

#### Evaluation Metrics

Use LLM-as-a-judge ([gemini-2.5-flash-lite](https://blog.google/products/gemini/gemini-2-5-model-family-expands/)) with the [Ragas](https://docs.ragas.io/en/stable/) module.

The metrics are as follows:

- [FactualCorrectness](https://docs.ragas.io/en/stable/references/metrics/#ragas.metrics.FactualCorrectness)
- [Faithfulness](https://docs.ragas.io/en/stable/references/metrics/#ragas.metrics.Faithfulness)
- [ContextRelevance](https://docs.ragas.io/en/stable/references/metrics/#ragas.metrics.ContextRelevance)

Additional resources for understanding the metrics:

- [RAG Evaluation Metrics Explained: A Complete Guide](https://freedium.cfd/https://medium.com/@med.el.harchaoui/rag-evaluation-metrics-explained-a-complete-guide-dbd7a3b571a8)

### Models

#### LLMs

- "Qwen/Qwen3-4B-Instruct-2507" [[Model Card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Paper](https://arxiv.org/pdf/2505.09388)]
- "meta-llama/Llama-3.2-3B-Instruct" [[Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [Blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)]
- "google/gemma-3-4b-it" [[Model Card](https://huggingface.co/google/gemma-3-4b-it), [Paper](https://arxiv.org/pdf/2503.19786)]

#### Embedding Models

- "Qwen/Qwen3-Embedding-0.6B" [[Model Card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Paper](https://arxiv.org/pdf/2506.05176)]
- "google/embeddinggemma-300m" [[Model Card](https://huggingface.co/google/embeddinggemma-300m), [Paper](https://arxiv.org/pdf/2509.20354)]

### RAG System

- Vector store: [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/), distance metric: cosine  
- Reranker: "colbert-ir/colbertv2.0" [[Model Card](https://huggingface.co/colbert-ir/colbertv2.0), [Paper](https://arxiv.org/pdf/2112.01488)]

### Decoding Strategies

- Decoding by Contrasting Layers (DoLa) [[Paper](https://arxiv.org/pdf/2309.03883), [GitHub](https://github.com/voidism/DoLa)]
- Self-Logits Evolution Decoding (SLED) [[Paper](https://arxiv.org/pdf/2411.02433), [GitHub](https://github.com/JayZhang42/SLED)]
- Activation Decoding [[Paper](https://arxiv.org/pdf/2403.01548), [GitHub](https://github.com/hkust-nlp/Activation_Decoding)]
- Entropy-eNhanced Decoding (END) [[Paper](https://arxiv.org/pdf/2502.03199), [GitHub](https://github.com/Arcade-Master/END)]

### Self-Reflection Prompting

[[Paper](https://arxiv.org/pdf/2310.06271), [GitHub](https://github.com/ziweiji/Self_Reflection_Medical)]

- **Knowledge Factuality Score** [[Paper](https://arxiv.org/pdf/2302.04166), [GitHub](https://github.com/jinlanfu/GPTScore)]:  
  By default, it uses paid APIs such as GPT or Gemini to evaluate the factuality score, but it is also possible to use the LLM model itself for this evaluation.

- **Answer Consistency Score** [[Paper](https://arxiv.org/pdf/2204.00862), [GitHub](https://github.com/thu-coai/CTRLEval)]:  
  By default, it uses "google/pegasus-large" [[Model Card](https://huggingface.co/google/pegasus-large)] to evaluate the consistency score.

- **QA Entailment Score**:  
  This metric measures sentence-BERT embedding similarity using the model "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" [[Model Card](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)].

## Experiments

Due to resource constraints, only 100 sample questions from test split of each dataset are used for experiments.

### Baselines

The LLM answers the questions directly.

The prompt template is as follows:

```python
"""Answer the following question in one paragraph.

Question: {question}
Answer:"""
```

Hyperparameters for answer generation (the same for all LLMs):

- `max_new_tokens` = 512
- `do_sample` = True
- `temperature` = 1.0
- `top_p` = 0.9
- `top_k` = 50
- `repetition_penalty` = 1.1

### Approaches to Beat the Baselines

1. Use different decoding strategies to reduce hallucinations and improve answer quality.
2. Use self-reflection prompting to reduce hallucinations and improve answer quality.
3. Use RAG with different LLMs and embedding models to improve answer quality based on retrieved context.

#### Experiment on Decoding Strategies

The LLM answers the questions using different decoding strategies (DoLa & SLED). The same prompt template as in the baselines is used.

Hyperparameters for answer generation (the same for all LLMs with decoding strategies):

- `max_new_tokens` = 512
- `do_sample` = False
- `repetition_penalty` = 1.2

For DoLa:

- `dola_layers` = 'high', which uses the hidden states of top half of the transformer layers to adjust the final logits.

For SLED: 

- `evolution_rate` = 2.0
- `evolution_scale` = 10

#### Experiment on Self-Reflection Prompting

The LLM answers the questions using self-reflection prompting.

Hyperparameters for answer generation (the same for all LLMs with self-reflection prompting):

- `max_new_tokens` = 512
- `do_sample` = True
- `temperature` = 1.0
- `top_p` = 0.9
- `top_k` = 50
- `repetition_penalty` = 1.1

Hyperparameters for the self-reflection loop:

- `max_loop` = 3
- `max_knowledge_loop` = 3
- `max_response_loop` = 3
- `threshold_entailment` = 0.8
- `threshold_fact` = -1.0
- `threshold_consistency` = -5.0

#### Experiment on RAG Systems

The LLM answers the questions using RAG systems with different LLMs and embedding models.

Three combinations of LLMs and embedding models are used:

1. Qwen/Qwen3-4B-Instruct-2507 + Qwen/Qwen3-Embedding-0.6B
2. meta-llama/Llama-3.2-3B-Instruct + Qwen/Qwen3-Embedding-0.6B
3. google/gemma-3-4b-it + google/embeddinggemma-300m

Documents from train, validation, and test splits are all used as the knowledge source for retrieval.

The prompt template is as follows:

```python
f"""You are a chatbot providing answers to user queries. You will be given one or more context documents, and a question. \
Use the information in the documents to answer the question.

If the documents do not provide enough information for you to answer the question, then say \
"The documents are missing some of the information required to answer the question." Don't quote any external knowledge that is \
not in the documents. Don't try to make up an answer.

Answer the question using the provided context.

Context:
{context}

Question: {question}"""
```

Hyperparameters for answer generation (the same for all RAG systems):

- `max_new_tokens` = 512
- `do_sample` = True
- `temperature` = 1.0
- `top_p` = 0.9
- `top_k` = 50
- `repetition_penalty` = 1.1

#### Experiment on RAG Systems with Different Decoding Strategies

The LLM answers the questions using RAG systems with different decoding strategies.

Only use RAG system with "Qwen/Qwen3-4B-Instruct-2507" + "Qwen/Qwen3-Embedding-0.6B" is used. Five combinations of RAG system and decoding strategies are as follows:

- RAG + DoLa
- RAG + SLED
- RAG + (DoLa + Activation): applying Activation Decoding on top of DoLa, with an alpha value balancing the effect of Activation Decoding on DoLa.
- RAG + (SLED + Activation): applying Activation Decoding on top of SLED, with an alpha value balancing the effect of Activation Decoding on SLED.
- RAG + (SLED + END): applying END on top of SLED, with an alpha value balancing the effect of END on SLED.

For the first set of experiments, hyperparameters for answer generation (the same for all RAG systems with different decoding strategies):

- `max_new_tokens` = 512
- `do_sample` = False
- `repetition_penalty` = 1.2

For DoLa:

- `dola_layers` = 'high'

For SLED:

- `evolution_rate` = 2.5
- `evolution_scale` = 75

For DoLa + Activation:

- `dola_layers` = 'high'
- `alpha` = 0.5

For SLED + Activation:

- `evolution_rate` = 0.5
- `evolution_scale` = 75
- `alpha` = 0.5

For SLED + END:

- `evolution_rate` = 0.5
- `evolution_scale` = 75
- `alpha` = 0.5

By default, the proposed decoding strategies by default using greedy decoding, so `do_sample` is set to False. Another set of experiments is conducted with `do_sample` set to True to see how these decoding strategies perform under sampling-based decoding.
For the second set of experiments, hyperparameters for answer generation (the same for all RAG systems with different decoding strategies):

- `max_new_tokens` = 512
- `do_sample` = True
- `temperature` = 1.0
- `top_p` = 0.9
- `top_k` = 50
- `repetition_penalty` = 1.2

The other hyperparameters for different decoding strategies are the same as above.

#### Experiment on RAG System with Self-Reflection Prompting

The LLM answers the questions using RAG system with self-reflection prompting.

Only the RAG system with "Qwen/Qwen3-4B-Instruct-2507" + "Qwen/Qwen3-Embedding-0.6B" is used. For this experiment, the self-reflection is applied not on the retrieved context but on the generated answer. This means the first part of the original self-reflection implementation is skipped.

Hyperparameters for answer generation:

- `max_new_tokens` = 512
- `do_sample` = True
- `temperature` = 1.0
- `top_p` = 0.9
- `top_k` = 50
- `repetition_penalty` = 1.1

Hyperparameters for self-reflection loop:

- `max_loop` = 3
- `max_response_loop` = 3
- `threshold_entailment` = 0.8
- `threshold_consistency` = -5.0