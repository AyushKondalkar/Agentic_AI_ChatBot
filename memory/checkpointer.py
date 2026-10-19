
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = Path(__file__).resolve().parents[1] / "chatbot_memory.db"


def get_checkpointer():
    # Keep the SQLite connection open for the app's lifetime.
    connection = SqliteSaver.from_conn_string(str(DB_PATH))
    checkpointer = connection.__enter__()

    return checkpointer, connection


def delete_thread_checkpoints(checkpointer, thread_id: str):
    """
    Delete all SQLite checkpoint records belonging to one thread.
    Other threads are left untouched.
    """
    conn = checkpointer.conn
    lock = getattr(checkpointer, "lock", None)

    if lock is not None:
        lock.acquire()

    try:
        conn.execute("BEGIN IMMEDIATE")

        # Find checkpoint tables containing a thread_id column.
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()

        for (table_name,) in tables:
            # Table names come from SQLite's own schema.
            quoted_table = '"' + table_name.replace('"', '""') + '"'

            columns = conn.execute(
                f"PRAGMA table_info({quoted_table})"
            ).fetchall()

            column_names = {column[1] for column in columns}

            if "thread_id" in column_names:
                conn.execute(
                    f"DELETE FROM {quoted_table} WHERE thread_id = ?",
                    (thread_id,),
                )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        if lock is not None:
            lock.release()
