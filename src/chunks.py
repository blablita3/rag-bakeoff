from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pprint import pprint
from src.data_utils import load_pdf_text, load_html


PAPER_PDF_PATH = Path("./data/refrag.pdf")
PAPER_HTML_PATH = Path("./data/refrag.html")
EMB_MODEL_NAME = "all-MiniLM-L6-v2"


# --- Chunking Helper Functions ---

def save_chunks_file(chunks, summary, save_dir):
    """
    Saves chunks (list of Documents) into jsonl, and a chunks summary
    (dict) into json. Returns saving paths.
    """
    method = summary["method"]

    # Build directory if necessary
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set paths
    chunks_save_path = save_dir / f"{method}_chunks.jsonl"
    summary_save_path = save_dir / f"{method}_summary.json"

    # Save chunks
    with open(chunks_save_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                json_obj = {
                    "metadata": chunk.metadata,
                    "page_content": chunk.page_content
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")   
    
    # Save summary
    with open(summary_save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return chunks_save_path, summary_save_path


def save_chunks_vector_db(chunks, save_dir, embedding_model):
    """
    Builds a Crroma db for storing the chunks (list of Documents).
    Requires an embedding model.
    Returns the vecorstore (Croma Obj) and the collection name (str).
    """

    # Build directory if necessary
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract source
    collection_name = chunks[0].metadata["method"]

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=save_dir
    )

    return vectorstore, collection_name 


def load_chunks_file(directory, method):
    """
    Loads and returns the chunks (list of Documents) stored as jsonl in the directory.
    """
    path = directory / f"{method}_chunks.jsonl"

    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                docs.append(
                    Document(
                        metadata=chunk["metadata"],
                        page_content=chunk["page_content"]
                    )
                )
    return docs


def load_chunks_db(directory, method, embedding_model):
    """
    Loads chunks (list of Documents) form a croma db at directory with the specified collecion name.
    Requires an embedding model.
    """
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name=method,
        persist_directory=directory
    )

    return vectorstore
    

def summarize_chunks(chunks, method):
    """
    Returns a summary (dict) of the given chunks (list of Documents)
    """
    lengths = [len(chunk.page_content) for chunk in chunks]

    summary = {
        "method": method,
        "num_chunks": len(chunks),
        "avg_length": np.mean(lengths),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "lengths_examples": lengths[:10]
        }
 
    return summary


# --- Different Chunking Methods ---

def recursive_chunking(paper_text, chunk_size=1000, chunk_overlap=200):
    """
    Recursively chunks the text input (str).
    Returns list of chunks as Documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split into chunks
    pre_chunks = splitter.split_text(paper_text)
    final_chunks = []

    for id, c in enumerate(pre_chunks):

        # Add metadata to the chunks
        final_chunks.append(Document(
            metadata = {
                "id": f"recursive_chunk_{id:03d}",
                "chunk_length": len(c),
                "method": "recursive"
            },
            page_content=c
        ))

    return final_chunks


def paragraph_chunking(html, min_paragraph_length = 100, merge_short_paragraphs = True, 
                       max_chunk_length = 2000, chunk_overlap = 200):
    
    """
    Does paragraph based chunking to an html (str) and recursively splits large paragraphs.
    Returns list of chunks as Documents.
    """

    # Tags we will extract
    block_tags = [
        "p", "div", "li", "paragraph", "article",
        "blockquote", "pre", "figcaption",
        "h1", "h2", "h3", "h4", "h5", "h6", "td"
    ]


    soup = BeautifulSoup(html, "lxml")

    # Remove noisy tags
    for t in soup(["script", "style", "math", "nav", "footer", "header", "form"]):
        t.decompose()
    body = soup.body or soup

    # Find candidate blocks in document order
    candidates = body.find_all(block_tags)

    # Keep only top-level blocks: skip an element if any ancestor (except body/html) is also a BLOCK_TAG
    top_blocks = []
    for el in candidates:
        skip = False
        for anc in el.parents:
            if anc is body:
                break
            if getattr(anc, "name", None) and anc.name.lower() in block_tags:
                skip = True
                break
        if not skip:
            top_blocks.append(el)

    # Extract paragraphs from each top-level block
    raw_paragraphs = []
    for el in top_blocks:

        # ensure <br> is treated as newline
        for br in el.find_all("br"):
            br.replace_with("\n")

        text = el.get_text(separator="\n", strip=True)
        if not text:
            continue

        # Normalize repeated newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Now split the block on blank lines (2+ newlines)
        parts = re.split(r"\n\s*\n", text)

        for part in parts:
            p = part.strip()
            if not p:
                continue

            # Join internal single-line wraps into spaces, but keep paragraphs separated
            p = re.sub(r"(?<!\n)\n(?!\n)", " ", p)
            p = re.sub(r"[ \t]{2,}", " ", p).strip()

            raw_paragraphs.append((p, el.name.lower()))

    # Merge short paragraphs into previous paragraph if they small
    paras_merged = []
    for text, tag in raw_paragraphs:
        if merge_short_paragraphs and paras_merged and len(text) < min_paragraph_length:
            
            # Append to previous with a space
            prev_text, prev_tag = paras_merged[-1]
            merged = prev_text + " " + text
            paras_merged[-1] = (merged.strip(), prev_tag)
        else:
            paras_merged.append((text, tag))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_length,
        chunk_overlap=chunk_overlap,
    )

    documents = []
    idx = 0
    for para_idx, (para_text, tagname) in enumerate(paras_merged):

        # Quit small paragraphs
        if len(para_text) < min_paragraph_length:
            continue
        
        if len(para_text) > max_chunk_length:
            
            # Divide large paragraphs
            small_chunks = splitter.split_text(para_text)
            for j, s in enumerate(small_chunks):

                # Remove splitted small chunks
                if len(s.strip()) < min_paragraph_length:
                    continue

                # Add metadata
                documents.append(Document(
                    page_content=s,
                    metadata={
                        "id": f"paragraph_chunk_{idx:03d}",
                        "paragraph_index": para_idx,
                        "split_index": j,
                        "orig_tag": tagname,
                        "length": len(s),
                        "method": "paragraph"
                    }
                ))
                idx += 1
        else:
            
            # Add metadata
            documents.append(Document(
                page_content=para_text,
                metadata={
                    "id": f"paragraph_chunk_{idx:03d}",
                    "paragraph_index": para_idx,
                    "split_index": 0,
                    "orig_tag": tagname,
                    "length": len(para_text),
                    "method": "paragraph"
                }
            ))
            idx += 1

    return documents


# Chunking methods: name, function and their paper format needed.

CHUNKING_METHODS = [("recursive", recursive_chunking, "text"), 
                    ("paragraph", paragraph_chunking, "html")]


# --- Main Chunking Functions ---

def build_method_chunks(method, method_chunking_func, paper_string, embedding_model, saving_dir):
    """ 
    Builds and saves chunks for a given method (str). Takes a chunking function for
    that method (callable), the paper as a str (may be html formatted) and an embedding model.

    Returns the chunks (list(Documents)) and the vectorstore (Croma Obj).
    """

    print(f"Building {method} chunks ...")
    
    # Chunk the paper
    chunks = method_chunking_func(paper_string)

    # Chunks summary
    summary = summarize_chunks(chunks, method)

    print(f"Summary for {method} chunking:")
    pprint(summary)

    save_dir = saving_dir / method

    # Save chunks as files
    chunks_path, summary_path = save_chunks_file(chunks, summary, save_dir)
    print(f"{method} chunks saved to {chunks_path}, summary saved to {summary_path}")

    # Save chunks as vector database
    vectorstore, collection_name = save_chunks_vector_db(chunks, save_dir, embedding_model)
    print(f"{method} chunks saved to Chroma DB at {save_dir}, with collection name '{collection_name}'\n")
    
    return chunks, vectorstore


def build_all_methods_chunks(saving_dir, chunking_methods=CHUNKING_METHODS, emb_model_name=EMB_MODEL_NAME, paper_pdf_path=PAPER_PDF_PATH, paper_html_path=PAPER_HTML_PATH):

    """
    Builds chunks for all chunking methods. 
    
    Takes chunking_methods a list of tuples. Each tuple contains what's needed for a chunking method:
    (method (str), chunking_function (callable), paper_format_needed (str)).
    Takes an embedding model name, and paths from the paper both as pdf and html.

    Returns chunks (dict(list(Documents))) a dict with the chunks for each chunking method,
    and vstores (dict(Croma VS Obj)) a dict with the vectorstores for each chunking method.
    """

    # Load paper on both formats
    paper = {
    "text": load_pdf_text(paper_pdf_path),
    "html": load_html(paper_html_path)
    }

    embedding_model = HuggingFaceEmbeddings(model_name=emb_model_name)
    
    print("Building chunks for all methods ...\n")

    all_chunks = {}
    all_vstores = {}

    # Build chunks for each method
    for method, function, format in chunking_methods:
        all_chunks[method], all_vstores[method] = build_method_chunks(method, function, paper[format], embedding_model, saving_dir)

    print("All chunking methods built\n")

    return all_chunks, all_vstores


def load_all_chunks(saving_dir, chunking_methods=CHUNKING_METHODS, emb_model_name=EMB_MODEL_NAME):
    """
    Loads all saved chunks from jsonl files and all chroma databases.
    Takes chunking methods (see builder for type), and saving directory path.
    """

    all_chunks = {}
    all_vstores = {}

    embedding_model = HuggingFaceEmbeddings(model_name=emb_model_name)

    # Load each method
    for method, _, _ in chunking_methods:

        # Chunks directory
        dir = saving_dir / method

        all_chunks[method] = load_chunks_file(dir, method)
        print(f"Loaded {len(all_chunks[method])} {method} chunks into Document objects")

        all_vstores[method] = load_chunks_db(dir, method, embedding_model)
        print(f"Loaded {method} chunks vectorstore\n")

    return all_chunks, all_vstores
