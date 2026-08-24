import argparse
import asyncio
import httpx
import json

async def main() -> int:
    p = argparse.ArgumentParser(description="SEC RAG Thin Client")
    p.add_argument("query", help="User question for the SEC RAG API")
    p.add_argument("--url", default="http://localhost:8000/ask", help="API Endpoint URL")
    args = p.parse_args()

    async with httpx.AsyncClient(timeout=300.0) as c:
        try:
            r = await c.post(args.url, json={"query": args.query})
        except httpx.RequestError as e:
            print(f"[X] Request Error: {e}")
            return 1
            
    if r.status_code != 200:
        print(f"[X] HTTP {r.status_code}: {r.text}")
        return 1
        
    data = r.json()
    print(f"\n{data['answer']}")
    print("\n--- Plan ---")
    print(json.dumps(data["sub_queries"], indent=2))
    print("\n--- Telemetry ---")
    print(json.dumps(data["telemetry"], indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))