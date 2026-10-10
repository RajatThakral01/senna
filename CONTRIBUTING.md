# Contributing to Viral Clips Automator

Thank you for your interest in contributing to **Viral Clips Automator**! We welcome contributions, bug reports, and suggestions.

---

## 🛠️ Development Setup

1. **Fork and Clone** the repository:
   ```bash
   git clone https://github.com/your-username/viral-clips.git
   cd viral-clips
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up PostgreSQL & pgvector** (Docker, port 5433 — see README step 4):
    ```bash
    docker run -d --name viral-clips-db -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=viral_clips \
      -p 5433:5432 -v viral_clips_pgdata:/var/lib/postgresql/data pgvector/pgvector:pg16
    db/init_db.sh --docker viral-clips-db
    ```

5. **Configure Environment Variables**:
    Copy `.env.example` to `.env` and fill in your Groq API key (LLM only — embeddings run locally) and database connection details:
   ```bash
   cp .env.example .env
   ```

---

## 🚀 Submitting Changes

1. **Branching**:
   Create a descriptive branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Code Style**:
   - Write clean, documented Python adhering to PEP 8 standards.
   - Do not commit secrets, tokens, or personal API keys.
   - Make sure runtime folders (`downloads/`, `work/`, `output/`, `logs/`, `models/`) are not committed (they are git-ignored).
   - Never hard-code `transcripts/`, `clips/` or `output/` paths — use `pipeline.workspace.current()`.
   - New `config.yaml` top-level sections must be added to the whitelist in `config.py:get_config()`.
   - Run the tests before pushing: `.venv/bin/python -m pytest -q` (300 tests; needs the DB).
   - Framing changes: compare `tools/framing_audit.py <video_id>` before/after and check the
     `--debug-framing` videos. Read `AGENT_CONTEXT.md` (architecture) and `OPEN_ISSUES.md` first.

3. **Commit Messages**:
   Follow conventional commits:
   - `feat: add support for local SRT subtitle styles`
   - `fix: resolve aspect ratio calculation in clipper`
   - `docs: update setup steps in README`

4. **Pull Requests**:
   - Push your branch to GitHub.
   - Open a Pull Request with a clear description of changes and motivation.

---

## 🐛 Reporting Issues

If you find a bug or have a feature request:
- Search existing GitHub Issues before opening a new one.
- Provide step-by-step instructions to reproduce the bug.
- Include OS version, Python version, FFmpeg version, and relevant error logs.
