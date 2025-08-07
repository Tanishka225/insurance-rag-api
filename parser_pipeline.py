
import re
import json
import os
from tqdm import tqdm
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\&\%\$\#\@]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def chunk_clauses(text: str) -> List[Dict[str, Any]]:
    section_pattern = re.compile(r'^(?P<section_number>\d+)\.\s+(?P<section_title>[^\n:]+)', re.MULTILINE)
    subclause_pattern = re.compile(r'^(?P<clause_label>[a-z]\))\s*', re.MULTILINE)
    sections, current_section = [], None
    lines, i = text.split("\n"), 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if (m := section_pattern.match(line)):
            if current_section: sections.append(current_section)
            current_section = {
                "section_number": m.group("section_number"),
                "section_title": m.group("section_title").strip(),
                "clauses": [], "word_count": 0
            }
            i += 1
            continue

        if (m := subclause_pattern.match(line)) and current_section:
            clause_text = line[len(m.group(0)):].strip()
            j = i + 1
            while j < len(lines) and not (section_pattern.match(lines[j]) or subclause_pattern.match(lines[j])):
                clause_text += " " + lines[j].strip()
                j += 1
            clause_obj = {
                "clause_label": m.group("clause_label"),
                "text": clause_text,
                "word_count": len(clause_text.split()),
                "char_count": len(clause_text)
            }
            current_section["clauses"].append(clause_obj)
            current_section["word_count"] += clause_obj["word_count"]
            i = j
            continue

        elif current_section:
            clause_text, j = line, i + 1
            while j < len(lines) and not (section_pattern.match(lines[j]) or subclause_pattern.match(lines[j])):
                clause_text += " " + lines[j].strip()
                j += 1
            clause_obj = {
                "clause_label": None,
                "text": clause_text,
                "word_count": len(clause_text.split()),
                "char_count": len(clause_text)
            }
            current_section["clauses"].append(clause_obj)
            current_section["word_count"] += clause_obj["word_count"]
            i = j
            continue
        i += 1

    if current_section:
        sections.append(current_section)

    if not sections:
        print("No structured sections found, using fallback paragraph chunking...")
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        fallback_section = {
            "section_number": "1",
            "section_title": "Document Content",
            "clauses": [], "word_count": 0
        }
        for i, p in enumerate(paragraphs):
            if len(p.split()) > 10:
                fallback_section["clauses"].append({
                    "clause_label": f"para_{i+1}",
                    "text": p,
                    "word_count": len(p.split()),
                    "char_count": len(p)
                })
                fallback_section["word_count"] += len(p.split())
        if fallback_section["clauses"]:
            sections.append(fallback_section)
    return sections

def create_rag_chunks(sections: List[Dict], max_chunk_size: int = 512) -> List[Dict[str, Any]]:
    chunks, chunk_id = [], 0
    for section in sections:
        section_context = f"Section {section['section_number']}: {section['section_title']}"
        for clause in section["clauses"]:
            if clause["word_count"] <= max_chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": clause["text"],
                    "metadata": {
                        **clause,
                        "context": section_context,
                        "section_number": section["section_number"],
                        "section_title": section["section_title"]
                    }
                })
                chunk_id += 1
            else:
                words = clause["text"].split()
                overlap = max_chunk_size // 4
                for i in range(0, len(words), max_chunk_size - overlap):
                    chunk_words = words[i:i + max_chunk_size]
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": " ".join(chunk_words),
                        "metadata": {
                            "word_count": len(chunk_words),
                            "char_count": len(" ".join(chunk_words)),
                            "context": section_context,
                            "section_number": section["section_number"],
                            "section_title": section["section_title"],
                            "clause_label": clause.get("clause_label"),
                            "is_split": True,
                            "split_index": i // (max_chunk_size - overlap)
                        }
                    })
                    chunk_id += 1
    return chunks

def smart_extract(path):
    try:
        if path.endswith(".pdf"):
            from unstructured.partition.pdf import partition_pdf
            return "\n".join(str(el) for el in partition_pdf(filename=path))
        elif path.endswith(".docx"):
            from unstructured.partition.docx import partition_docx
            return "\n".join(str(el) for el in partition_docx(filename=path))
        elif path.endswith(".eml"):
            from unstructured.partition.email import partition_email
            return "\n".join(str(el) for el in partition_email(filename=path))
        else:
            print(f"Unsupported file type: {path}")
            return None
    except Exception as e:
        print(f"Failed to extract from {path}: {e}")
        return None

def process_documents_for_rag(folder_path: str, output_file: str = "rag_ready_chunks.json"):
    print("Step 1: Extracting text from documents...")
    extracted_data = []
    for file in tqdm(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            text = smart_extract(full_path)
            if text:
                extracted_data.append({"filename": file, "text": text})
                print(f"Successfully extracted from: {file}")
            else:
                print(f"Failed to extract from: {file}")
    print("Step 2: Cleaning and chunking text...")
    all_chunks = []
    for item in tqdm(extracted_data):
        cleaned = clean_text(item["text"])
        sections = chunk_clauses(cleaned)
        chunks = create_rag_chunks(sections)
        for chunk in chunks:
            chunk["metadata"]["source_file"] = item["filename"]
            all_chunks.append(chunk)
    print(f"Step 3: Saving {len(all_chunks)} chunks to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print("Processing Complete!")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Files processed: {len(extracted_data)}")
    return all_chunks
