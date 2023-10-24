# Dolla Llama: Real-Time Co-Pilot for Closing the Deal 📞

<p align="center">
  <img src="assets/dolla_llama.png" alt="Dolla Llama" style="width:30%;height:30%">
</p>

This assistant implements speech-to-text (STT) and retrieval-augmented generation (RAG) to assist live sales calls. 

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

Simply launch with:
```
docker-compose up
```

And navigate to `http://localhost:8090` 

## Creating Custom Embeddings

## Indexing with ElasticSearch

## Interface with Gradio

## Next Steps

* Fine-tune an LLM for your usecase
* Add additional indices for query/retrieval
* Try a container orchestrator like k8s for robust distributed deployments
