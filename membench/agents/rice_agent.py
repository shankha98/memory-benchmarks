import json
import os
import sys
import uuid
from queue import Empty, Queue
from threading import Thread
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import AliasChoices, BaseModel, Field
from ricedb import RiceDBClient


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: str | None = None


class InputCase(BaseModel):
    case_id: str
    input: str = Field(validation_alias=AliasChoices("input", "question"))
    history: list[Message] = Field(default_factory=list)
    choices: dict[str, str] = Field(default_factory=dict)


def _create_rice_client(
    *,
    host: str,
    port: int,
    password: str,
    ssl: bool,
) -> RiceDBClient:
    # RiceDBClient prints to stdout during connection; keep benchmark output clean.
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        rice_client = RiceDBClient(host, port=port)
        rice_client.ssl = ssl
        if not rice_client.connect():
            raise RuntimeError("Failed to connect to RICE server")
        rice_client.login("admin", password)
        return rice_client
    finally:
        sys.stdout = old_stdout


def _drain_errors(error_queue: "Queue[BaseException]") -> list[BaseException]:
    drained: list[BaseException] = []
    while True:
        try:
            drained.append(error_queue.get_nowait())
        except Empty:
            return drained


def main():
    # pacabench may set env vars to empty strings when `${VAR}` is unset; treat that
    # as "unset" so dotenv can populate them.
    if os.environ.get("RICE_PASSWORD", "") == "":
        os.environ.pop("RICE_PASSWORD", None)

    run_id = str(uuid.uuid4())

    dotenv_path = os.getenv("DOTENV_PATH") or find_dotenv(usecwd=False)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    proxy_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

    rice_host = os.getenv("RICE_HOST", "api.ricedb-beta-m5xd9.ricedb.tryrice.com")
    rice_port = int(os.getenv("RICE_PORT", "80"))
    rice_password = (os.getenv("RICE_PASSWORD") or "").strip()
    rice_ssl = os.getenv("RICE_SSL", "false").lower() == "true"

    if not rice_password:
        error = "Missing RICE_PASSWORD (set env var or put it in .env)"
        sys.stderr.write(error + "\n")
        for line in sys.stdin:
            if not line.strip():
                continue
            print(json.dumps({"output": None, "error": error}))
            try:
                sys.stdout.flush()
            except BrokenPipeError:
                return
        return

    try:
        main_client = _create_rice_client(
            host=rice_host,
            port=rice_port,
            password=rice_password,
            ssl=rice_ssl,
        )
    except Exception as e:
        error = f"Failed to initialize RICE client: {e}"
        sys.stderr.write(error + "\n")
        for line in sys.stdin:
            if not line.strip():
                continue
            print(json.dumps({"output": None, "error": error}))
            try:
                sys.stdout.flush()
            except BrokenPipeError:
                return
        return

    openai_client = OpenAI(api_key=api_key, base_url=proxy_url)
    node_counter = 0

    insert_workers = max(1, int(os.getenv("RICE_INSERT_WORKERS", "2")))
    insert_batch_size = max(1, int(os.getenv("RICE_INSERT_BATCH_SIZE", "64")))

    insert_queue: Queue[object] | None = None
    insert_errors: "Queue[BaseException]" | None = None
    stop_sentinel: object | None = None
    worker_clients: list[RiceDBClient] = []
    worker_threads: list[Thread] = []

    def start_insert_workers() -> None:
        nonlocal insert_queue, insert_errors, stop_sentinel
        if insert_queue is not None:
            return

        insert_queue = Queue()
        insert_errors = Queue()
        stop_sentinel = object()

        try:
            for _ in range(insert_workers - 1):
                worker_clients.append(
                    _create_rice_client(
                        host=rice_host,
                        port=rice_port,
                        password=rice_password,
                        ssl=rice_ssl,
                    )
                )
        except Exception as e:
            sys.stderr.write(f"Failed to initialize RICE insert workers: {e}\n")
            sys.exit(1)

        def insert_worker(worker_client: RiceDBClient) -> None:
            assert insert_queue is not None
            assert insert_errors is not None
            assert stop_sentinel is not None
            while True:
                item = insert_queue.get()
                try:
                    if item is stop_sentinel:
                        return
                    batch_user_id, documents = item  # type: ignore[misc]
                    worker_client.batch_insert(documents, user_id=batch_user_id)
                except Exception as e:
                    insert_errors.put(RuntimeError(f"RICE batch_insert failed: {e}"))
                finally:
                    insert_queue.task_done()

        for worker_client in worker_clients:
            thread = Thread(target=insert_worker, args=(worker_client,), daemon=True)
            thread.start()
            worker_threads.append(thread)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            case = InputCase.model_validate(data)
        except Exception:
            continue

        user_id = abs(hash(case.case_id)) % (10**9)

        try:
            documents: list[dict] = []
            start_node_id = node_counter + 1
            node_counter += len(case.history)
            for idx, msg in enumerate(case.history):
                documents.append(
                    {
                        "id": start_node_id + idx,
                        "text": msg.content,
                        "metadata": {
                            "role": msg.role,
                            "index": idx,
                            "run_id": run_id,
                            "text": msg.content,
                        },
                    }
                )

            batches = [
                documents[offset : offset + insert_batch_size]
                for offset in range(0, len(documents), insert_batch_size)
            ]

            if insert_workers > 1 and len(batches) > 1:
                start_insert_workers()
                assert insert_queue is not None
                assert insert_errors is not None
                case_errors = _drain_errors(insert_errors)
                if case_errors:
                    raise case_errors[0]

                main_thread_batches: list[list[dict]] = []
                for batch_index, batch in enumerate(batches):
                    if batch_index % insert_workers == 0:
                        main_thread_batches.append(batch)
                    else:
                        insert_queue.put((user_id, batch))

                for batch in main_thread_batches:
                    main_client.batch_insert(batch, user_id=user_id)

                insert_queue.join()
                case_errors = _drain_errors(insert_errors)
                if case_errors:
                    raise case_errors[0]
            else:
                for batch in batches:
                    main_client.batch_insert(batch, user_id=user_id)

            # Search
            search_results = main_client.search(
                case.input, user_id=user_id, k=20, filter={"run_id": run_id}
            )

            # Extract messages - need to re-fetch content since search only returns metadata
            retrieved_messages = []
            if search_results:
                for result in search_results:
                    if isinstance(result, dict):
                        metadata = result.get("metadata", {})
                        node_id = result.get("node_id")
                        role = metadata.get("role", "user")
                        index = metadata.get("index", 0)
                        # Get text from metadata, fallback to history
                        content = metadata.get("text")
                        if not content and 0 <= index < len(case.history):
                            content = case.history[index].content

                        if content:
                            retrieved_messages.append(
                                {"role": role, "content": content, "index": index}
                            )

            retrieved_messages.sort(key=lambda x: x.get("index", 0))

            # Build prompt
            context_text = "Relevant conversation history:\n"
            for msg in retrieved_messages:
                context_text += f"[{msg['role']}]: {msg['content']}\n"

            user_prompt = f"{context_text}\nQuestion: {case.input}"
            if case.choices:
                choices_text = "\n".join(
                    f"{k}. {v}" for k, v in sorted(case.choices.items())
                )
                user_prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Use the provided conversation context to answer the question.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
            )
            try:
                print(
                    json.dumps(
                        {"output": response.choices[0].message.content, "error": None}
                    )
                )
            except BrokenPipeError:
                return

        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            try:
                print(json.dumps({"output": None, "error": str(e)}))
            except BrokenPipeError:
                return

        try:
            sys.stdout.flush()
        except BrokenPipeError:
            return

    try:
        if worker_threads:
            assert insert_queue is not None
            assert stop_sentinel is not None
            for _ in worker_threads:
                insert_queue.put(stop_sentinel)
            insert_queue.join()
            for thread in worker_threads:
                thread.join(timeout=5)

            for worker_client in worker_clients:
                worker_client.disconnect()

        main_client.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
