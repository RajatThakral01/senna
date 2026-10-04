"""
db/connection.py

PostgreSQL connection pool for the Viral Clips Automator.

Input:  Database credentials from config.yaml (merged with DB_USER / DB_PASSWORD env vars).
Output: psycopg2 connection objects drawn from a ThreadedConnectionPool.

Usage:
    from db.connection import get_conn, release_conn

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    finally:
        release_conn(conn)
"""
import os
import psycopg2
from psycopg2 import pool
import logging

_logger = logging.getLogger("db.connection")

# pgvector registers its vector type with psycopg2 automatically on import
try:
    from pgvector.psycopg2 import register_vector
    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False

from config import get_config

_pool = None


def get_pool():
    """Lazily initialise the ThreadedConnectionPool and return it."""
    global _pool
    if _pool is None:
        cfg = get_config()["database"]
        _logger.debug("init pool host=%s port=%s db=%s user=%s", cfg["host"], cfg.get("port"), cfg["name"], cfg["user"])
        conn_kwargs = dict(
            minconn=1,
            maxconn=10,
            host=cfg["host"],
            port=cfg.get("port", 5432),
            dbname=cfg["name"],
            user=cfg["user"],
        )
        # Only pass password if one is set (macOS trust auth often has no password)
        if cfg.get("password"):
            conn_kwargs["password"] = cfg["password"]

        try:
            _pool = psycopg2.pool.ThreadedConnectionPool(**conn_kwargs)
        except Exception:
            _logger.exception("pool init failed host=%s db=%s", cfg["host"], cfg["name"])
            raise
        _logger.info("pool ready host=%s db=%s", cfg["host"], cfg["name"])
    return _pool


def get_conn():
    """Get a connection from the pool. Caller must call release_conn() when done."""
    conn = get_pool().getconn()
    if _PGVECTOR_AVAILABLE:
        register_vector(conn)
    return conn


def release_conn(conn):
    """Return a connection to the pool."""
    get_pool().putconn(conn)
