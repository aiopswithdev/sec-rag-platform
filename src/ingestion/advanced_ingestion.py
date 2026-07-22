"""
Production-grade SEC 10-K ingestion with:
- Dual-action state machine to prevent data bleed from non-whitelisted Items
- DOM de-duplication via table markers + NavigableString iteration
- Single-pass table processing (validation + heuristic summary)
- Deterministic UUIDs, SQLite parent store, Qdrant hybrid vector store
"""

import os
import re
import uuid
import sqlite3
import logging
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(">>> [ENV] Loaded environment variables from local .env file.")
except ImportError:
    print(">>> [ENV] python-dotenv not installed. Falling back to system environment.")

print(">>> [INIT] Importing Pandas...")
import pandas as pd

print(">>> [INIT] Importing NLTK...")
import nltk
# Download the required sentence tokenizers for NLTKTextSplitter
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print(">>> [INIT] Importing BeautifulSoup...")
from bs4 import BeautifulSoup, NavigableString, Tag

print(">>> [INIT] Importing Qdrant...")
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

print(">>> [INIT] Importing FastEmbed...")
from fastembed import TextEmbedding, SparseTextEmbedding

print(">>> [INIT] Importing Langchain Text Splitters...")
from langchain_text_splitters import NLTKTextSplitter

print(">>> [INIT] Importing TQDM...")
from tqdm import tqdm

print(">>> [INIT] ALL IMPORTS SUCCESSFUL. Moving to main()...")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

NAMESPACE_SEC = uuid.uuid5(uuid.NAMESPACE_DNS, "sec.gov")

# ---------------------------------------------------------------------------
# Database & Collection Validation
# ---------------------------------------------------------------------------
def setup_sqlite():
    conn = sqlite3.connect("parent_docstore.db")
    cursor = conn.cursor()

    # Enable WAL mode for high-concurrency read/write operations
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parent_documents (
            id TEXT PRIMARY KEY,
            ticker TEXT,
            fiscal_year INTEGER,
            item_number TEXT,
            full_text TEXT,
            is_table BOOLEAN
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_parent_lookup
        ON parent_documents(ticker, fiscal_year, item_number)
    """)
    conn.commit()
    return conn


def validate_qdrant_collection(client: QdrantClient, collection_name: str):
    """Enforces infrastructure boundaries by verifying the schema exists before parsing."""
    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' does not exist! "
            "You must execute 'initialize_collection.py' to establish the schema and payload indexes first."
        )
    logging.info(f"Verified: Target collection '{collection_name}' is active with pre-built indices.")


# ---------------------------------------------------------------------------
# Fiscal Year Extraction
# ---------------------------------------------------------------------------
def extract_fiscal_year(raw_html: str, accession_num: str = "") -> int:
    for line in raw_html.splitlines():
        if 'FILED AS OF DATE' in line.upper():
            match = re.search(r'\b(\d{4})\b', line)
            if match:
                return int(match.group(1))
    for pattern in [
        r'<FILED-AS-OF-DATE>\s*(\d{4})',
        r'<PERIOD-END>\s*(\d{4})',
        r'<CONFORMED-PERIOD-OF-REPORT>\s*(\d{4})',
    ]:
        m = re.search(pattern, raw_html, re.IGNORECASE)
        if m:
            return int(m.group(1))
    if accession_num:
        m = re.search(r'-(\d{2})-', accession_num)
        if m:
            return 2000 + int(m.group(1))
    logging.warning("Fiscal year not found, defaulting to 2024")
    return 2024


# ---------------------------------------------------------------------------
# Table Processing (single pass)
# ---------------------------------------------------------------------------
def process_table(ticker, item_name, table_html):
    from io import StringIO
    try:
        df_list = pd.read_html(StringIO(table_html))
        if not df_list:
            return False, ""
        df = df_list[0]
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if df.shape[0] < 2 or df.shape[1] < 2:
            return False, ""

        row_labels = []
        for val in df.iloc[:, 0].dropna().tolist():
            if isinstance(val, str) and len(val.strip()) > 1:
                row_labels.append(val.strip())
            if len(row_labels) >= 8:
                break

        if row_labels:
            return True, f"Financial Table for {ticker} in {item_name}. Rows: {', '.join(row_labels)}."
        
        cols = list(df.columns.astype(str))
        if not all(col.isdigit() for col in cols):
            summary = f"Financial Table for {ticker} in {item_name}. Columns: {', '.join(cols[:8])}."
            first_row = df.iloc[0].tolist() if len(df) > 0 else []
            if first_row:
                summary += f" Sample row: {', '.join(str(v) for v in first_row[:8])}."
        else:
            flat_values = []
            for row_idx in range(min(2, df.shape[0])):
                flat_values.extend(df.iloc[row_idx].dropna().tolist()[:4])
            sample = ", ".join(str(v) for v in flat_values[:8])
            summary = f"Financial Table for {ticker} in {item_name}. Sample values: {sample}."
        return True, summary.strip()
    except Exception:
        return False, ""


# ---------------------------------------------------------------------------
# Section Extraction with Dual-Action State Machine & Table Markers
# ---------------------------------------------------------------------------
def extract_whitelisted_items(html_content: str):
    soup = BeautifulSoup(html_content, 'lxml')
    for tag in soup(['script', 'style', 'head', 'meta', 'link']):
        tag.decompose()

    heading_re = re.compile(
        r'^Item\s+([1-9][A-Z]?)(?:[.\-:\s]|$)', 
        re.IGNORECASE | re.DOTALL
    )

    tables_with_headings = {}
    for table in soup.find_all('table'):
        full_text = table.get_text(' ', strip=True)
        m = heading_re.match(full_text)
        if m and len(m.group()) < 250:
            tables_with_headings[table] = m.group()

    table_map = {}
    table_counter = 0
    for table in soup.find_all('table'):
        # 1. Detachment Defense: Skip if this table was already removed 
        # (e.g., it was an inner child of a previously replaced table).
        if table.parent is None:
            continue
            
        # 2. Layout Wrapper Defense: SEC filings often use tables for page padding.
        # If this table contains another table, it is a structural layout grid.
        # Skip markering the outer wrapper so the parser safely reaches the inner data matrix.
        if table.find('table'):
            continue

        marker_id = f"__TABLE_{table_counter}__"
        table_map[marker_id] = str(table)
        
        if table in tables_with_headings:
            heading_text = tables_with_headings[table]
            text_node = NavigableString(heading_text + '\n')
            table.insert_before(text_node)
            
        # Safe replacement of the verified data container
        marker = soup.new_tag('span', attrs={'class': 'table-marker', 'data-id': marker_id})
        table.replace_with(marker)
        table_counter += 1

    order_map = {
        '1': 1, '1A': 2, '1B': 3, '2': 4, '3': 5, '4': 6,
        '5': 7, '6': 8, '7': 9, '7A': 10, '8': 11, '9': 12,
        '9A': 13, '9B': 14, '10': 15, '11': 16, '12': 17,
        '13': 18, '14': 19, '15': 20
    }
    whitelist = {'1A', '7', '8'}
    sections = {}
    current_item_key = None
    last_seen_order = -1
    recording = False
    JUMP_THRESHOLD = 8

    def is_heading(text, element):
        if element.parent and element.parent.get('class') == ['table-marker']:
            return False
        if not heading_re.match(text):
            return False
        if len(text) > 250:
            return False
        if re.search(r'\brefer\b|\bsee\b|\bcontain', text, re.IGNORECASE):
            return False
        return True

    body = soup.find('body') or soup
    for node in body.descendants:
        if isinstance(node, Tag) and 'table-marker' in node.get('class', []):
            if recording and current_item_key:
                marker_id = node['data-id']
                if marker_id in table_map:
                    sections[current_item_key].append({'type': 'table', 'content': table_map[marker_id]})
            continue

        if not isinstance(node, NavigableString):
            continue

        text = node.strip()
        if len(text) < 3:
            continue

        if is_heading(text, node):
            match = heading_re.match(text)
            item_num = match.group(1).upper()
            item_order = order_map.get(item_num, 999)

            if last_seen_order >= 0 and (item_order - last_seen_order) > JUMP_THRESHOLD:
                continue

            if item_order > last_seen_order:
                last_seen_order = item_order
                if item_num in whitelist:
                    current_item_key = f'Item {item_num}'
                    if current_item_key not in sections:
                        sections[current_item_key] = []
                    recording = True
                    sections[current_item_key].append({'type': 'text', 'content': text})
                else:
                    current_item_key = None
                    recording = False
            continue

        if recording and current_item_key:
            sections[current_item_key].append({'type': 'text', 'content': text})

    for item, elements in sections.items():
        merged = []
        buf = []
        for el in elements:
            if el['type'] == 'text':
                buf.append(el['content'])
            else:
                if buf:
                    merged.append({'type': 'text', 'content': '\n'.join(buf)})
                    buf = []
                merged.append(el)
        if buf:
            merged.append({'type': 'text', 'content': '\n'.join(buf)})
        sections[item] = merged

    return sections


# ---------------------------------------------------------------------------
# Main Ingestion Loop
# ---------------------------------------------------------------------------
def main():
    print(">>> [1/5] Initializing SQLite Docstore...")
    sqlite_conn = setup_sqlite()
    sqlite_cursor = sqlite_conn.cursor()

    print(">>> [2/5] Connecting to Qdrant...")
    api_key = os.getenv("QDRANT_API_KEY")
    
    # Standardized to HTTP port 6333 to match the API service layer
    qdrant_client = QdrantClient(
        host="localhost", 
        port=6334, 
        api_key=api_key,
        prefer_grpc=True,
        https=False
    )
    
    collection_name = "advanced_sec_edgar_production" 
    validate_qdrant_collection(qdrant_client, collection_name)

    print(">>> [3/5] Loading Dense Embedding Model...")
    t0 = time.time()
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print(f"    -> Dense model loaded in {time.time() - t0:.1f} seconds.")
    
    print(">>> [4/5] Loading Sparse BM25 Model...")
    t0 = time.time()
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    print(f"    -> Sparse model loaded in {time.time() - t0:.1f} seconds.")
    
    child_splitter = NLTKTextSplitter(chunk_size=400, chunk_overlap=50)

    print(">>> [5/5] INFRASTRUCTURE READY. Scanning local directories...")
    base_dir = "./data/sec-edgar-filings"
    BATCH_SIZE = 100
    points_batch = []
    
    file_paths = []
    for ticker in os.listdir(base_dir):
        ticker_path = os.path.join(base_dir, ticker, "10-K")
        if not os.path.isdir(ticker_path):
            continue
        for accession_num in os.listdir(ticker_path):
            file_path = os.path.join(ticker_path, accession_num, "full-submission.txt")
            if os.path.exists(file_path):
                file_paths.append((ticker, accession_num, file_path))

    total_files = len(file_paths)
    print(f"\n==================================================")
    print(f"STARTING PRODUCTION INGESTION: {total_files} FILES")
    print(f"==================================================\n")
    
    global_start_time = time.time()

    for ticker, accession_num, file_path in tqdm(file_paths, desc="Ingesting 10-Ks", unit="file"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_html = f.read()

            fiscal_year = extract_fiscal_year(raw_html, accession_num)
            item_sections = extract_whitelisted_items(raw_html)

            for item_name, elements in item_sections.items():
                parent_text_blocks = []
                table_index = 0

                for el in elements:
                    if el["type"] == "text":
                        parent_text_blocks.append(el["content"])
                    elif el["type"] == "table":
                        raw_table_html = el["content"]
                        is_valid, summary = process_table(ticker, item_name, raw_table_html)
                        if not is_valid:
                            continue

                        table_index += 1
                        parent_id = str(uuid.uuid5(
                            NAMESPACE_SEC,
                            f"{ticker}_{fiscal_year}_{item_name}_table_{table_index}"
                        ))

                        sqlite_cursor.execute(
                            "INSERT OR REPLACE INTO parent_documents "
                            "(id, ticker, fiscal_year, item_number, full_text, is_table) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (parent_id, ticker, fiscal_year, item_name, raw_table_html, True)
                        )

                        dense_vec = list(dense_model.embed([summary]))[0].tolist()
                        sparse_result = list(sparse_model.embed([summary]))[0]

                        points_batch.append(
                            PointStruct(
                                id=str(uuid.uuid5(NAMESPACE_SEC, f"vec_{parent_id}")),
                                vector={
                                    "dense-text": dense_vec,
                                    "sparse-text": {
                                        "indices": sparse_result.indices.tolist(),
                                        "values": sparse_result.values.tolist()
                                    }
                                },
                                payload={
                                    "parent_id": parent_id,
                                    "ticker": ticker,
                                    "fiscal_year": fiscal_year,
                                    "item_number": item_name,
                                    "is_table": True,
                                    "text": summary,
                                    "llm_enriched": False
                                }
                            )
                        )
                        if len(points_batch) >= BATCH_SIZE:
                            qdrant_client.upsert(collection_name=collection_name, points=points_batch)
                            points_batch = []
                # [FIX 1: EXPLICIT DOMAIN FLUSH]
                # Flush immediately after table processing to protect high-value vectors from prose exceptions
                if points_batch:
                    qdrant_client.upsert(collection_name=collection_name, points=points_batch)
                    points_batch = []

                # Now safely proceed to prose processing
                full_prose = "\n".join(parent_text_blocks)
                if len(full_prose) < 500:
                    continue

                parent_id = str(uuid.uuid5(
                    NAMESPACE_SEC,
                    f"{ticker}_{fiscal_year}_{item_name}_prose"
                ))
                sqlite_cursor.execute(
                    "INSERT OR REPLACE INTO parent_documents "
                    "(id, ticker, fiscal_year, item_number, full_text, is_table) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (parent_id, ticker, fiscal_year, item_name, full_prose, False)
                )

                child_chunks = child_splitter.split_text(full_prose)
                for i, child_text in enumerate(child_chunks):
                    dense_vec = list(dense_model.embed([child_text]))[0].tolist()
                    sparse_result = list(sparse_model.embed([child_text]))[0]
                    points_batch.append(
                        PointStruct(
                            id=str(uuid.uuid5(NAMESPACE_SEC, f"vec_{parent_id}_chunk_{i}")),
                            vector={
                                "dense-text": dense_vec,
                                "sparse-text": {
                                    "indices": sparse_result.indices.tolist(),
                                    "values": sparse_result.values.tolist()
                                }
                            },
                            payload={
                                "parent_id": parent_id,
                                "ticker": ticker,
                                "fiscal_year": fiscal_year,
                                "item_number": item_name,
                                "is_table": False,
                                "text": child_text
                            }
                        )
                    )
                    if len(points_batch) >= BATCH_SIZE:
                        qdrant_client.upsert(collection_name=collection_name, points=points_batch)
                        points_batch = []
                # FIX: Explicitly flush remaining prose vectors per SEC Item
                if points_batch:
                    qdrant_client.upsert(collection_name=collection_name, points=points_batch)
                    points_batch = []

        except Exception as e:
            logging.error(f"Failed processing {ticker} {accession_num}: {e}", exc_info=True)
            continue

    if points_batch:
        qdrant_client.upsert(collection_name=collection_name, points=points_batch)

    sqlite_conn.commit()
    sqlite_conn.close()
    
    total_time = time.time() - global_start_time
    files_per_sec = total_files / total_time if total_time > 0 else 0
    print(f"\n==================================================")
    print(f"PIPELINE COMPLETE.")
    print(f"Processed {total_files} files in {total_time:.2f} seconds ({files_per_sec:.2f} files/sec).")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()