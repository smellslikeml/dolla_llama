# Dolla Llama: Real-Time Co-Pilot for Closing the Deal

<p align="center">
  <img src="assets/dolla_llama.png" alt="Dolla Llama" style="width:30%;height:30%">
</p>

Implements speech-to-text (STT) and retrieval-augmented generation (RAG) to assist live sales calls. 

## 🌟 Features:
- STT with Whisper.cpp and llama.cpp for your LLM
- Custom embeddings for your text corpus using SentenceTransformers
- Indexing documents + embeddings with ElasticSearch

## Table of Contents
1. [Getting Started](#getting-started)
2. [Creating Custom Embeddings](#creating-custom-embeddings)
3. [Indexing with ElasticSearch](#indexing-with-elasticsearch)
4. [Interface with Gradio](#interface-with-gradio)
5. [Next Steps](#next-steps)

## Getting Started
This demo assumes you have:

- [docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/) installed
- Familiarity with [RAG](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/) and its applications

### Setup
Make sure to convert your Llama model with [llama.cpp](https://github.com/ggerganov/llama.cpp) for serving using these [instructions]().
Simply launch with:
```
docker-compose up
```

And navigate to `http://localhost:8090` 

## Creating Custom Embeddings

By fine-tuning with SentenceTransformers, we can generate text embeddings locally for matching with documents in our Elasticsearch index.

The [scraper/main.py](scraper/main.py) script scrapes a list of sites to index. You can update the links in `scraper/config.json` 

## Indexing with ElasticSearch

Using Elasticsearch, we can index and tag documents for filtering and customization of the relevance scoring.

The [scraper/main.py](scarper/main.py) script also handles this after scraping. 

## Interface with Gradio

With Gradio, you press a button to begin and read suggestions in the chatbox.

The [app/app.py](app/app.py) contains the logic to run whisper for speech-to-text, run queries on the elasticsearch index, and launch the front-end. 

## Next Steps

* Fine-tune an LLM for your usecase
* Add additional indices for query/retrieval
* Try a container orchestrator like k8s for robust distributed deployments
