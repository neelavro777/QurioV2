# Qurio

Qurio is an AI assistant that suggests YouTube videos, podcasts, articles, and other resources to help you keep up with the rapid development of AI.

Recommendations are tailored to specific categories like:
- Vision Language Models (VLMs)
- Convex
- Next.js
- Reinforcement Learning
- ...and more!

---

## 🛠️ Python Environment Setup (Ubuntu / Linux)

Since Qurio is set up using the modern fast python package manager **`uv`**, you have two main ways to interact with your environment. Both the modern **`uv`** workflows and traditional **`pip`** workflows are documented below.

### 1. Activating/Sourcing the Virtual Environment (`venv`)

To activate the virtual environment created in your project directory:

```bash
# Sourcing the virtual environment from the project root
source .venv/bin/activate
```

*To deactivate the environment when you're done, simply run:*
```bash
deactivate
```

> [!TIP]
> **With `uv`, you don't even need to activate the virtual environment!**
> You can run scripts directly using `uv run <command>`. For example:
> `uv run python main.py`
> This automatically runs your code in the isolated environment with all project dependencies available.

---

### 2. Managing Dependencies & `requirements.txt`

#### 💡 The Modern `uv` Way (Recommended)
Since this project has a `pyproject.toml` initialized, you can add, remove, and manage packages directly via the `uv` toolchain:

*   **Add package(s) to the project:**
    ```bash
    uv add langchain langgraph pydantic python-dotenv langchain-community
    ```
    *(This automatically updates `pyproject.toml` and updates the virtual environment).*

*   **Generate/Export a `requirements.txt` from project definition:**
    ```bash
    uv export --format requirements-txt -o requirements.txt
    ```

*   **Install packages from an existing `requirements.txt`:**
    ```bash
    uv pip install -r requirements.txt
    ```

*   **Sync the virtual environment with `requirements.txt` (removes extra packages, installs missing ones):**
    ```bash
    uv pip sync requirements.txt
    ```

---

#### 🐍 The Traditional `pip` Way
If you prefer standard python toolsets or need to work outside `uv`:

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```
2. **Install from a `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Save current dependencies to `requirements.txt`:**
   ```bash
   pip freeze > requirements.txt
   ```

---

## 🚀 Running the Project

To execute the entry point of the project:

```bash
uv run python main.py
```
