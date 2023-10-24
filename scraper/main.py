import os
import json
import sys
import scrapy
from time import sleep
from scrapy.crawler import CrawlerProcess
from sentence_transformers import (
    SentenceTransformer,
    SentencesDataset,
    LoggingHandler,
    losses,
)
from torch.utils.data import DataLoader
from elasticsearch import Elasticsearch


def wait_for_elasticsearch(es_host):
    es = Elasticsearch(hosts=es_host)
    while True:
        try:
            if es.ping():
                return
        except:
            pass
        sleep(5)


class TextSpider(scrapy.Spider):
    name = "text_spider"
    with open("config.json", "r") as f:
        config = json.load(f)
        start_urls = config.get("start_urls", [])

    custom_settings = {
        "DEPTH_LIMIT": 3,  # Adjust as needed
        "DOWNLOAD_DELAY": 1,  # Adjust delay between requests
    }

    def parse(self, response):
        yield {"url": response.url, "text": response.xpath("//text()").extract()}
        for next_page in response.css("a::attr(href)").extract():
            yield response.follow(next_page, self.parse)


def save_texts_to_files(texts):
    # Save the texts into separate files
    for idx, text_data in enumerate(texts):
        with open(f"text_{idx}.txt", "w") as f:
            f.write("\n".join(text_data["text"]))


def index_to_elastic(texts, embeddings):
    es = Elasticsearch()

    for idx, text_data in enumerate(texts):
        doc = {
            "url": text_data["url"],
            "text": "\n".join(text_data["text"]),
            "embedding": embeddings[idx].tolist(),
        }
        es.index(index="text_embeddings", doc_type="_doc", body=doc)


def main():
    es_host = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

    # Wait for Elasticsearch to be ready
    wait_for_elasticsearch(es_host)

    # Scrapy extraction
    process = CrawlerProcess()
    process.crawl(TextSpider)
    texts = list(process.start())

    save_texts_to_files(texts)

    # Training model
    model = SentenceTransformer("paraphrase-distilroberta-base-v2")
    sentences = ["\n".join(text_data["text"]) for text_data in texts]
    train_dataset = SentencesDataset(sentences, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)])
    model.save("/models/embedding_model")

    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Index to Elasticsearch
    index_to_elastic(texts, embeddings)


if __name__ == "__main__":
    main()
