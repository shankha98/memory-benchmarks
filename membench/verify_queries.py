import re
import sys
import uuid
import os
from dotenv import find_dotenv, load_dotenv
from ricedb import RiceDBClient

# Queries to verify
QUERIES = {
    "simple-2": "How large is Logistics Pro's scale?",
    "simple-4": "How long does the Delivery Summit last?",
    "simple-6": "How long does TechSpark 2024 last?",
    "simple-9": "What time does Zen Beach Bash start?",
}


def parse_logs(log_path):
    cases = {}
    current_case = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            # Match "Inserting ... for case simple-X:"
            match = re.match(r"Inserting \d+ documents for case (simple-\d+):", line)
            if match:
                current_case = match.group(1)
                cases[current_case] = []
                continue

            if current_case and line.startswith("- "):
                text = line[2:]  # Remove "- "
                cases[current_case].append(text)

            if line.startswith("--------------------"):
                current_case = None

    return cases


def main():
    load_dotenv(find_dotenv(usecwd=False))

    # Check if we are in membench directory
    if os.path.exists("rice_agent_debug.log"):
        log_path = "rice_agent_debug.log"
    elif os.path.exists("membench/rice_agent_debug.log"):
        log_path = "membench/rice_agent_debug.log"
    else:
        print(f"Log file not found in . or membench/")
        return

    print("Parsing logs...")
    case_data = parse_logs(log_path)

    rice_host = os.getenv("RICE_HOST", "api.ricedb-beta-m5xd9.ricedb.tryrice.com")
    rice_port = int(os.getenv("RICE_PORT", "80"))
    rice_password = (os.getenv("RICE_PASSWORD") or "").strip()
    rice_ssl = os.getenv("RICE_SSL", "false").lower() == "true"

    if not rice_password:
        print("Error: RICE_PASSWORD not set.")
        return

    print(f"Connecting to RiceDB at {rice_host}:{rice_port}...")
    try:
        client = RiceDBClient(rice_host, port=rice_port)
        client.ssl = rice_ssl
        if not client.connect():
            print("Failed to connect to RICE server")
            return
        client.login("admin", rice_password)
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    for case_id, query in QUERIES.items():
        print(f"\nVerifying {case_id}...")
        texts = case_data.get(case_id)
        if not texts:
            print(f"No data found for {case_id} in logs.")
            continue

        print(f"Found {len(texts)} documents for {case_id}.")

        run_id = str(uuid.uuid4())
        user_id = abs(hash(case_id)) % (10**9)

        documents = []
        for idx, text in enumerate(texts):
            documents.append(
                {
                    "id": idx + 1,  # Simple ID
                    "text": text,
                    "metadata": {
                        "role": "user",  # Placeholder
                        "index": idx,
                        "run_id": run_id,
                        "text": text,
                    },
                }
            )

        print(f"Inserting documents (run_id={run_id})...")
        try:
            client.batch_insert(documents, user_id=user_id)
        except Exception as e:
            print(f"Insertion failed: {e}")
            continue

        print(f"Searching for: '{query}'...")
        try:
            results = client.search(
                query, user_id=user_id, k=5, filter={"run_id": run_id}
            )
            if results:
                print("Results found:")
                for res in results:
                    meta = res.get("metadata", {})
                    print(f" - {meta.get('text', 'NO_TEXT')}")
            else:
                print("NO RESULTS FOUND.")

        except Exception as e:
            print(f"Search failed: {e}")

    client.disconnect()


if __name__ == "__main__":
    main()
