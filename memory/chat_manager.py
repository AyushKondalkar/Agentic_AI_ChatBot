
import json
import uuid
from pathlib import Path


class ChatManager:
    def __init__(self):
        self.threads_file = (
            Path(__file__).resolve().parents[1]
            / "chat_threads.json"
        )

    def load_threads(self):
        """Load all saved chat threads."""

        if not self.threads_file.exists():
            return []

        try:
            with open(
                self.threads_file,
                "r",
                encoding="utf-8",
            ) as file:
                threads = json.load(file)

            return threads if isinstance(threads, list) else []

        except (json.JSONDecodeError, OSError):
            return []

    def save_threads(self, threads):
        """Save chat thread metadata."""

        with open(
            self.threads_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                threads,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def create_thread(self):
        """Create a new chat with a unique ID."""

        thread = {
            "id": str(uuid.uuid4()),
            "title": "New Chat",
        }

        threads = self.load_threads()
        threads.insert(0, thread)

        self.save_threads(threads)

        return thread["id"]

    def update_thread_title(self, thread_id, message):
        """Set the title using the first user message."""

        threads = self.load_threads()

        for thread in threads:
            if thread["id"] == thread_id:
                if thread["title"] == "New Chat":
                    thread["title"] = message.strip()[:40]
                break

        self.save_threads(threads)
