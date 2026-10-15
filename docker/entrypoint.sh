#!/usr/bin/env bash
# Wait for Postgres, apply schema + migrations (idempotent), then run the command.
set -euo pipefail
export PGPASSWORD="${DB_PASSWORD:-}"
for i in $(seq 1 60); do
  if pg_isready -q -h "${DB_HOST:-db}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}"; then
    break
  fi
  echo "waiting for the database (${i}s)…"; sleep 1
done
db/init_db.sh
mkdir -p downloads output work logs models assets/logos assets/music assets/campaigns media
exec "$@"
