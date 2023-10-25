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
    InputExample,
)
from torch.utils.data import DataLoader
from elasticsearch import Elasticsearch
from urllib.parse import urlparse



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
    items = []
    with open("config.json", "r") as f:
        config = json.load(f)
        start_urls = config.get("start_urls", [])

    custom_settings = {
        "DEPTH_LIMIT": 1,  # Setting to 1 ensures only the start URLs are processed
        "DOWNLOAD_DELAY": 1,  # Adjust delay between requests
        # Other custom settings to prevent blocks or restrictions
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
        "DOWNLOAD_TIMEOUT": 30,  # You can adjust this as per your needs
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,  # Total number of retries for each request
        "COOKIES_ENABLED": False,  # Disabling cookies as some sites might track based on cookies
        # If the site uses AJAX, you might consider enabling the middlewares below:
        # 'scrapy.downloadermiddlewares.ajaxcrawl.AjaxCrawlMiddleware': 400,
    }

    def parse(self, response):
        item = {"url": response.url, "text": response.xpath("//text()").extract()}
        self.items.append(item) 
        yield item


def save_texts_to_files(texts):
    # Save the texts into separate files
    for idx, text_data in enumerate(texts):
        with open(f"text_{idx}.txt", "w") as f:
            f.write("\n".join(text_data["text"]))


def index_to_elastic(texts, embeddings, es_host):
    es = Elasticsearch(hosts=[es_host])

    # Ensure the index exists before inserting documents
    try:
        if not es.indices.exists(index="text_embeddings"):
            es.indices.create(index="text_embeddings")
    except NotFoundError:
        es.indices.create(index="text_embeddings")

    for idx, text_data in enumerate(texts):
        doc = {
            "url": text_data["url"],
            "text": "\n".join(text_data["text"]),
            "embedding": embeddings[idx].tolist(),
        }
        es.index(index="text_embeddings",body=doc)


def main():
    es_host = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

    # Wait for Elasticsearch to be ready
    wait_for_elasticsearch(es_host)

    # Scrapy extraction
    process = CrawlerProcess()
    process.crawl(TextSpider)
    process.start()

    texts = TextSpider.items  # This will fetch the scraped items


    save_texts_to_files(texts)

    # Training model
    model = SentenceTransformer("paraphrase-distilroberta-base-v2")
    # Prepare sentences for encoding
    sentences = ["\n".join(text_data["text"]) for text_data in texts]

    # Encode the sentences directly
    embeddings = model.encode(sentences)

    # Index to Elasticsearch
    index_to_elastic(texts, embeddings, es_host)



if __name__ == "__main__":
    main()
