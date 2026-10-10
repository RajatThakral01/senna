#!/usr/bin/env bash
# Create the pgvector extension, base schema and migrations v4-v7 (idempotent).
#
# Usage:
#   db/init_db.sh                         # psql via env: DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME
#   db/init_db.sh --docker viral-clips-db # run psql inside the Docker container
set -euo pipefail
cd "$(dirname "$0")"

DB_NAME="${DB_NAME:-viral_clips}"
DB_USER="${DB_USER:-postgres}"

if [ "${1:-}" = "--docker" ]; then
  CONTAINER="${2:-viral-clips-db}"
  run() { docker exec -i "$CONTAINER" psql -v ON_ERROR_STOP=1 -q -U "$DB_USER" -d "$DB_NAME" "$@"; }
else
  export PGPASSWORD="${DB_PASSWORD:-}"
  run() { psql -v ON_ERROR_STOP=1 -q -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" \
               -U "$DB_USER" -d "$DB_NAME" "$@"; }
fi

run -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pgcrypto;"
for f in schema.sql migration_v4.sql migration_v5.sql migration_v6.sql migration_v7.sql; do
  echo "applying $f"
  run < "$f"
done
echo "database ready: $DB_NAME"
