
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import tempfile
import os

from model_rag import SimpleRAGPipeline, assign_unique_ids
from parser_pipeline import smart_extract, clean_text, chunk_clauses, create_rag_chunks

import torch
from getpass import getpass

app = FastAPI()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

rag = SimpleRAGPipeline(pinecone_api_key=pinecone_api_key, index_name="finserv-rag")


# Request schema
class RAGRequest(BaseModel):
    documents: str
    questions: List[str]

# Response schema
class RAGResponse(BaseModel):
    answers: List[str]


@app.post("/hackrx/run", response_model=RAGResponse)
def run_rag_query(request: RAGRequest):
    try:
        # Step 1: Download PDF file
        doc_url = request.documents
        response = requests.get(doc_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # Step 2: Extract + clean + chunk
        raw_text = smart_extract(tmp_file_path)
        cleaned = clean_text(raw_text)
        sections = chunk_clauses(cleaned)
        chunks = create_rag_chunks(sections)
        chunks = assign_unique_ids(chunks, namespace="runtime")

        # Step 3: Embed + upload to Pinecone temporarily
        embedded_chunks = rag.create_embeddings(chunks)
        rag.upload_to_pinecone(embedded_chunks)

        # Step 4: Answer each question
        answers = []
        for question in request.questions:
            result = rag.query(question)
            answers.append(result["answer"])

        return RAGResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
