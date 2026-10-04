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

4. **Set Up PostgreSQL & pgvector**:
   Ensure PostgreSQL is installed and running, then execute the schema:
   ```bash
   createdb viral_clips
   psql viral_clips -f db/schema.sql
   ```

5. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your NVIDIA NIM credentials and database connection details:
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
   - Make sure runtime output folders (`output/`, `clips/`, `downloads/`, `transcripts/`) are kept clean and not committed.

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
