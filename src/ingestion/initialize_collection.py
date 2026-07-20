import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PayloadSchemaType
from dotenv import load_dotenv

def setup_hybrid_collection():
    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        raise ValueError("QDRANT_API_KEY environment variable is missing.")

    client = QdrantClient(
        host="localhost",
        port=6334,
        api_key=api_key,
        prefer_grpc=True,
        https=False
    )

    collection_name = "advanced_sec_edgar_production"

    # 1. Collection Provisioning
    if not client.collection_exists(collection_name):
        print(f"Creating hybrid collection '{collection_name}'...")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense-text": VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse-text": SparseVectorParams()
                }
            )
            print(f"[-] Collection '{collection_name}' successfully initialized.")
        except Exception as e:
            print(f"[X] Failed to create collection: {e}")
            return # Abort if we couldn't create the base collection
    else:
        print(f"[-] Collection '{collection_name}' already exists. Proceeding to schema validation.")

    # 2. Payload Schema Indexing (Executes whether the collection is new or pre-existing)
    print("[-] Registering structural payload schema indices...")
    try:
        client.create_payload_index(collection_name, "ticker", PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, "fiscal_year", PayloadSchemaType.INTEGER)
        client.create_payload_index(collection_name, "item_number", PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, "is_table", PayloadSchemaType.BOOL)
        print("[-] Payload indices registered successfully.")
    except Exception as e:
        print(f"[X] Failed to register payload indices: {e}")

    # 3. Final Verification
    status = client.get_collection(collection_name)
    print(f"[-] Status: {status.status}")

if __name__ == "__main__":
    load_dotenv()
    setup_hybrid_collection()