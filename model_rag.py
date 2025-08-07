
import json
import numpy as np
import time
from typing import List, Dict, Any
from tqdm import tqdm
import torch
import gc
from getpass import getpass

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from getpass import getpass
import os
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def assign_unique_ids(chunks: List[Dict], namespace: str = "default") -> List[Dict]:
    id_set = set()
    for i, chunk in enumerate(chunks):
        source = chunk['metadata'].get('source_file', 'unknown')
        section = chunk['metadata'].get('section_number', i)
        unique_id = f"{namespace}::{source}::{section}::{i}"
        if unique_id in id_set:
            raise ValueError(f"Duplicate ID detected even after formatting: {unique_id}")
        id_set.add(unique_id)
        chunk['chunk_id'] = unique_id
    return chunks

class SimpleRAGPipeline:
    def __init__(self, pinecone_api_key: str, index_name: str = "finserv-rag"):
        self.index_name = index_name
        self.embedding_dim = 384
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.pc = Pinecone(api_key=pinecone_api_key)
        self._setup_pinecone_index()
        self._setup_flan_t5()

    def _setup_pinecone_index(self):
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            if self.index_name in existing_indexes:
                print(f"Using existing index: {self.index_name}")
                self.index = self.pc.Index(self.index_name)
            else:
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(2)
                self.index = self.pc.Index(self.index_name)
                print("Index created and ready")
        except Exception as e:
            print(f"Error setting up Pinecone: {e}")
            raise

    def _setup_flan_t5(self):
        model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            self.generator = self.generator.to('cuda')
            print("FLAN-T5 loaded on GPU")
        else:
            print("FLAN-T5 loaded on CPU")

    def create_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32):
        texts = [chunk['text'] for chunk in chunks]
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
            if i % (batch_size * 4) == 0:
                gc.collect()
        enriched_chunks = []
        for chunk, embedding in zip(chunks, all_embeddings):
            enriched_chunk = chunk.copy()
            enriched_chunk['embedding'] = embedding.tolist()
            enriched_chunks.append(enriched_chunk)
        print(f"Created embeddings for {len(enriched_chunks)} chunks")
        return enriched_chunks

    def upload_to_pinecone(self, chunks_with_embeddings: List[Dict[str, Any]], batch_size: int = 100):
        print(f"Uploading {len(chunks_with_embeddings)} vectors to Pinecone...")
        vectors_to_upload = []
        for chunk in chunks_with_embeddings:
            vector = {
                'id': str(chunk['chunk_id']),
                'values': chunk['embedding'],
                'metadata': {
                    'text': chunk['text'][:1000],
                    'source_file': chunk['metadata']['source_file'],
                    'section_number': chunk['metadata']['section_number'],
                    'section_title': chunk['metadata']['section_title'][:100],
                    'word_count': chunk['metadata']['word_count']
                }
            }
            vectors_to_upload.append(vector)
        for i in tqdm(range(0, len(vectors_to_upload), batch_size), desc="Uploading to Pinecone"):
            batch = vectors_to_upload[i:i + batch_size]
            self.index.upsert(vectors=batch)
        time.sleep(3)
        stats = self.index.describe_index_stats()
        print(f"Upload complete! Index now has {stats['total_vector_count']} vectors")

    def search_similar_chunks(self, question: str, top_k: int = 5):
        question_embedding = self.embedder.encode([question])[0].tolist()
        search_results = self.index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        similar_chunks = []
        for match in search_results['matches']:
            similar_chunks.append({
                'id': match['id'],
                'similarity_score': match['score'],
                'text': match['metadata']['text'],
                'source_file': match['metadata']['source_file'],
                'section_number': match['metadata']['section_number'],
                'section_title': match['metadata']['section_title'],
                'word_count': match['metadata']['word_count']
            })
        return similar_chunks

    def generate_answer(self, question: str, similar_chunks: List[Dict], max_length: int = 300):
        context_parts = [f"Source {i+1}: {chunk['text'][:400]}" for i, chunk in enumerate(similar_chunks)]
        context = "\n\n".join(context_parts)
        prompt = (
            f"Answer the following question based on the provided context from financial services documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=5,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response.strip()
        return answer

    def query(self, question: str, top_k: int = 5):
        similar_chunks = self.search_similar_chunks(question, top_k)
        if not similar_chunks:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "sources": [],
                "num_sources": 0
            }
        answer = self.generate_answer(question, similar_chunks)
        sources = []
        for chunk in similar_chunks:
            sources.append({
                "file": chunk["source_file"],
                "section": f"Section {chunk['section_number']}: {chunk['section_title']}",
                "similarity": round(chunk["similarity_score"], 3)
            })
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }

def setup_rag_system():
    rag = SimpleRAGPipeline(
        pinecone_api_key=pinecone_api_key,
        index_name="finserv-rag-system"
    )
    return rag

def process_and_upload_chunks(rag, chunks_file_path: str):
    with open(chunks_file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    chunks = assign_unique_ids(chunks, namespace="insurance")
    chunks_with_embeddings = rag.create_embeddings(chunks, batch_size=32)
    rag.upload_to_pinecone(chunks_with_embeddings, batch_size=100)
    return chunks_with_embeddings
