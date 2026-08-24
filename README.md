# Northstar Commerce & Fitzy Sales Agent — V1 Release Candidate

Welcome to the **V1 Release Candidate** for the Northstar Commerce Platform & Fitzy Autonomous AI Sales Agent.

## Architectural Overview

- **Single Entrypoint**: `main.py` serves both the Northstar Commerce backend APIs and the Fitzy Sales Agent endpoint.
- **Single HTTP Port**: `8000` (Unified app mounting `/api/v1/products`, `/api/v1/carts`, `/api/v1/orders`, `/api/v1/agent/chat`, `/health`, `/health/ready`).
- **Authoritative LLM Client**: Uses `CLOTHING_AGENT_LLM_*` environment configuration (fallback `GROQ_API_KEY`) pointing to `openai/gpt-oss-120b`.
- **Multilingual Resilience**: Full support for English, Roman Urdu ("mujhe shadi ke liye kuch acha sa chahiye"), and Urdu script ("مجھے شادی کے لیے کچھ اچھا سا چاہیے").
- **Zero Frontend Breaking Changes**: Fully compatible with existing client apps.

## Quickstart

### 1. Environment Setup

Copy `.env.example` to `.env` and fill in your Groq or OpenAI API Key:

```bash
cp .env.example .env
```

Ensure `.env` contains:
```env
PORT=8000
CLOTHING_AGENT_LLM_API_KEY=your_groq_api_key_here
CLOTHING_AGENT_LLM_MODEL=openai/gpt-oss-120b
CLOTHING_AGENT_CLOTHING_APP_BASE_URL=http://127.0.0.1:8000
```

### 2. Run the Unified Service

```bash
python main.py
```
Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run the Verification Test Suites

To verify the commerce backend suite:
```bash
cd clothing_app
python -m pytest tests -q
```

To verify the Fitzy AI Agent release candidate suite (including live LLM end-to-end tests):
```bash
cd clothing_agent
python -m pytest tests -q
```
Or run the release candidate verification harness directly:
```bash
python -m pytest clothing_agent/tests/test_release_candidate.py -v -s
```

## Health Checks

- `/health`: Verifies PostgreSQL DB connection and LLM client configuration.
- `/health/ready`: Readiness probe verifying all internal dependencies.
