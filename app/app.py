import os
import sys
import openai
import guidance
import subprocess
import numpy as np
import logging
from ast import literal_eval
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from guidance import models, gen, select

import guidance
import gradio as gr
import subprocess

es_host = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
es = Elasticsearch(hosts=es_host)
llama_host = os.environ.get("LLAMA_URL", "http://localhost:8000")


model = SentenceTransformer("paraphrase-distilroberta-base-v2")


def embedding(text_input):
    return model.encode(text_input)

def search_es(query):
    query_embedding = embedding(query).tolist()  # Generate embedding for the query

    body = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['embedding']) + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        },
        "_source": ["url", "text"]  # Adjust the fields to return as per your needs
    }
    response = es.search(index="text_embeddings", body=body, size=10)  # Adjust 'size' as needed
    return response['hits']['hits']


# Load the LLM model
os.environ[
    "OPENAI_API_KEY"
] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # can be anything
os.environ["OPENAI_API_BASE"] = llama_host + "/v1"
os.environ["OPENAI_API_HOST"] = llama_host

llm_path = os.path.join(llama_host, "v1") 
llama2 = models.OpenAI("text-davinci-003", base_url=llm_path)

cmd = [
    "/whisper/stream",
    "-m",
    "/whisper/models/ggml-tiny.en-q5_0.bin",
    "--step",
    "7680",
    "--length",
    "15360",
    "-c",
    "0",
    "-t",
    "3",
    "-ac",
    "800",
]

# Start the process
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,  # Output as text instead of bytes
    bufsize=1,  # Line buffered (optional)
)


def process_audio(audio_chunk):
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as process:
        # Pass audio_chunk as input and get output
        stdout_data, stderr_output = process.communicate(input=audio_chunk)

    # Read lines from the process's stdout
    idx = 0
    conversation_buffer = []
    output_text = ""
    for line in stdout_data.splitlines():
        idx += 1

        conversation_buffer.append(line.strip())
        if len(conversation_buffer) > 5:
            transcript = " ".join(conversation_buffer)
            es_results = search_es(transcript)
            es_info = " ".join([hit['_source']['content'] for hit in es_results])

            response = (llama2 + f"Transcript: {transcript}\nElasticsearch Information: {es_info}\n" +
                       f"Respond with short, readable phrases to quickly prompt the conversation: " +
                       gen(stop='.')).strip()

            conversation_buffer.pop(0)

            print(response)
            output_text += response + "\n"

    # If there was an error, return that to the user
    if output_text == "":
        return f"Error processing audio: {stderr_output}"
    return output_text


iface = gr.Interface(
    process_audio,
    gr.Audio(source="microphone", streaming=True),
    "text",
    live=True,
    server_port=8090,
)
iface.launch()
