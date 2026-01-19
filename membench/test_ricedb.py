import os
import sys
import uuid
from dotenv import find_dotenv, load_dotenv
from ricedb import RiceDBClient


def main():
    # Load environment variables
    load_dotenv(find_dotenv(usecwd=False))

    rice_host = os.getenv("RICE_HOST", "api.ricedb-beta-m5xd9.ricedb.tryrice.com")
    rice_port = int(os.getenv("RICE_PORT", "80"))
    rice_password = (os.getenv("RICE_PASSWORD") or "").strip()
    rice_ssl = os.getenv("RICE_SSL", "false").lower() == "true"

    if not rice_password:
        print("Error: RICE_PASSWORD not set in environment or .env file.")
        sys.exit(1)

    print(f"Connecting to RiceDB at {rice_host}:{rice_port}...")

    try:
        client = RiceDBClient(rice_host, port=rice_port)
        client.ssl = rice_ssl
        if not client.connect():
            print("Failed to connect to RICE server")
            sys.exit(1)

        client.login("admin", rice_password)
        print("Connected and logged in.")

        # Test data
        user_id = 12345
        run_id = str(uuid.uuid4())
        print(f"Generated run_id: {run_id}")

        documents = [
            {
                "id": 1,
                "text": "The quick brown fox jumps over the lazy dog.",
                "metadata": {
                    "source": "test",
                    "text": "The quick brown fox jumps over the lazy dog.",
                    "run_id": run_id,
                },
            },
            {
                "id": 2,
                "text": "Python is a powerful programming language.",
                "metadata": {
                    "source": "test",
                    "text": "Python is a powerful programming language.",
                    "run_id": run_id,
                },
            },
            {
                "id": 3,
                "text": "RiceDB is a vector database.",
                "metadata": {
                    "source": "test",
                    "text": "RiceDB is a vector database.",
                    "run_id": run_id,
                },
            },
            {
                "id": 4,
                "text": "Artificial Intelligence is transforming the world.",
                "metadata": {
                    "source": "test",
                    "text": "Artificial Intelligence is transforming the world.",
                    "run_id": run_id,
                },
            },
            {
                "id": 5,
                "text": "Machine learning models require data.",
                "metadata": {
                    "source": "test",
                    "text": "Machine learning models require data.",
                    "run_id": run_id,
                },
            },
        ]

        print(f"Inserting {len(documents)} documents for user_id={user_id}...")
        client.batch_insert(documents, user_id=user_id)
        print("Insertion complete.")

        query = "What is RiceDB?"
        print(f"Searching for: '{query}' with run_id filter...")
        # Use run_id filter to ensure we only get results from this run
        results = client.search(query, user_id=user_id, k=2, filter={"run_id": run_id})

        print(f"Found {len(results) if results else 0} results:")
        if results:
            for i, result in enumerate(results):
                print(f"Result {i + 1}: {result}")
        else:
            print("No results found.")

        # Test run_id isolation
        wrong_run_id = str(uuid.uuid4())
        print(f"\nTesting isolation with wrong run_id={wrong_run_id}...")
        isolation_results = client.search(
            query, user_id=user_id, k=2, filter={"run_id": wrong_run_id}
        )
        if not isolation_results:
            print("SUCCESS: No results found for wrong run_id.")
        else:
            print(f"FAILURE: Found {len(isolation_results)} results for wrong run_id!")

        client.disconnect()
        print("Disconnected.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
