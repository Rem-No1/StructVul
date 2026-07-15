import json
import os
from http import HTTPStatus
from typing import Any, Dict, List

import chromadb
import dashscope
import requests
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm

from .cpg_utils import extract_cpg_structure


DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
# API-based option, though it may underperform qwen3-embedding-0.6b.
EMBEDDING_MODEL_NAME = "text-embedding-v3"
# Self-host qwen3-embedding-0.6b for better results at higher resource cost.
# EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"  


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class QwenEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = 5
        self.max_input_chars = 8000
        self._embedding_dim = None

    def _prepare_text(self, text: str) -> str:
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return " "
        if len(text) > self.max_input_chars:
            return text[: self.max_input_chars]
        return text

    def _embed_batch(self, batch_input: List[str]) -> List[List[float]]:
        resp = dashscope.TextEmbedding.call(
            model=self.model_name,
            input=batch_input,
            api_key=self.api_key,
        )

        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"Dashscope API Error: {resp.message}")

        embeddings = [item["embedding"] for item in resp.output["embeddings"]]
        if len(embeddings) != len(batch_input):
            raise RuntimeError(
                f"Embedding count mismatch: expect {len(batch_input)}, got {len(embeddings)}"
            )
        if embeddings and self._embedding_dim is None:
            self._embedding_dim = len(embeddings[0])
        return embeddings

    def __call__(self, input: Documents) -> Embeddings:
        prepared_input = [self._prepare_text(text) for text in input]
        all_embeddings: List[List[float]] = []

        for i in range(0, len(prepared_input), self.batch_size):
            batch_input = prepared_input[i : i + self.batch_size]
            try:
                all_embeddings.extend(self._embed_batch(batch_input))
            except Exception as batch_error:
                print(f"Batch embedding failed, retry each item: {batch_error}")
                for single_text in batch_input:
                    try:
                        all_embeddings.extend(self._embed_batch([single_text]))
                    except Exception as single_error:
                        print(f"Single embedding failed: {single_error}")
                        if self._embedding_dim is None:
                            raise RuntimeError(
                                "Embedding service unavailable; unable to infer embedding dimension."
                            ) from single_error
                        all_embeddings.append([0.0] * self._embedding_dim)

        if len(all_embeddings) != len(prepared_input):
            raise RuntimeError(
                f"Embedding generation incomplete: expect {len(prepared_input)}, got {len(all_embeddings)}"
            )
        return all_embeddings


class VulnRAG:
    def __init__(self, dataset_path: str, persist_dir: str = "./cpg_chroma_db", collection_name: str = "cpg_vuln_kb"):
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

        os.environ["ANONYMIZED_TELEMETRY"] = "False"

        self.client = chromadb.PersistentClient(path=persist_dir)
        dashscope_api_key = require_env(DASHSCOPE_API_KEY_ENV)
        self.ef = QwenEmbeddingAdapter(api_key=dashscope_api_key, model_name=EMBEDDING_MODEL_NAME)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            print(f"Building CPG knowledge base from {dataset_path} ...")
            self._ingest_data(dataset_path)
        else:
            print(f"Loaded CPG knowledge base: {self.collection.count()} entries")

    def _ingest_data(self, dataset_path: str) -> None:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []

        print("Extracting CPG features and building embeddings ...")
        for idx, item in enumerate(tqdm(data, desc="Processing CPG")):
            code = item.get("code", "")
            if not code:
                continue

            doc_id = str(item.get("id", f"idx_{idx}"))
            cpg_text = extract_cpg_structure(code)

            ids.append(doc_id)
            documents.append(cpg_text)
            metadatas.append(
                {
                    "original_code": code,
                    "is_vulnerable": str(item.get("label", 0)),
                    "vuln_type": str(item.get("vuln_type", "Unknown")),
                    "reason": str(item.get("reason", "No reason"))[:1000],
                }
            )

        batch_size = 10
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        print("CPG knowledge base build completed.")

    def rerank(self, query: str, retrieved_docs: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        if not retrieved_docs:
            return []

        payload = {
            "model": "qwen3-rerank",
            "input": {
                "query": query,
                "documents": [doc["code"] for doc in retrieved_docs],
            },
            "parameters": {
                "return_documents": False,
                "top_n": top_n,
            },
        }
        headers = {
            "Authorization": f"Bearer {require_env(DASHSCOPE_API_KEY_ENV)}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )
            if response.status_code != 200:
                print(f"Rerank failed: {response.text}")
                return retrieved_docs[:top_n]

            result = response.json()
            final_docs: List[Dict[str, Any]] = []
            for item in result["output"]["results"]:
                doc = dict(retrieved_docs[item["index"]])
                doc["rerank_score"] = item["relevance_score"]
                final_docs.append(doc)
            return final_docs
        except Exception as exc:
            print(f"Rerank error: {exc}")
            return retrieved_docs[:top_n]

    def search(
        self,
        query_code: str,
        k: int = 10,
        rerank_top_k: int = 3,
        exclude_id: str = None,
    ) -> List[Dict[str, Any]]:
        query_cpg = extract_cpg_structure(query_code)
        results = self.collection.query(
            query_texts=[query_cpg],
            n_results=k + 2,
        )

        candidates: List[Dict[str, Any]] = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                cid = str(results["ids"][0][i])
                if exclude_id and cid == str(exclude_id):
                    continue

                candidates.append(
                    {
                        "id": cid,
                        "code": results["metadatas"][0][i]["original_code"],
                        "is_vulnerable": results["metadatas"][0][i]["is_vulnerable"],
                        "vuln_type": results["metadatas"][0][i]["vuln_type"],
                        "reason": results["metadatas"][0][i]["reason"],
                        "vector_distance": results["distances"][0][i] if "distances" in results else 0,
                    }
                )

        return self.rerank(query=query_code, retrieved_docs=candidates, top_n=rerank_top_k)

    def search_similar_cases(self, query_code: str, k: int = 3, exclude_id: str = None) -> List[Dict[str, Any]]:
        return self.search(query_code=query_code, rerank_top_k=k, exclude_id=exclude_id)
