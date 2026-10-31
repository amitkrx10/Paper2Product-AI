# Paper2Product AI — an autonomous research-to-code engineering agent

> Upload a research paper. Get an implementation, a reproducible experiment, verified execution evidence,
> and an **honest** comparison with the reported results.

Hackathon track: **AI and Developer Tools**. Paper2Product is a tool-using agent that ingests a paper, plans,
resolves data, generates a project, executes it in a sandbox, reads the failures, repairs them within a budget,
measures metrics itself, and reports exactly what was and was not verified.

## What was actually verified (read this first)

| Area | Status | Evidence |
|---|---|---|
| Core engine: ingest → plan → data → codegen → sandbox → repair → evaluate → report | **Verified, executed** | 105 automated tests pass; CLI demo run end-to-end |
| PDF error handling (corrupt PDF with PyMuPDF installed → clean `IngestError`, pypdf fallback, scanned-PDF rejection) | **Fixed after external review**, regression-tested with a simulated PyMuPDF | `tests/test_ingest_schemas.py::TestPyMuPDFFailureHandling` |
| Repair loop (injected `NameError` → diagnose → patch → rerun) | **Verified, executed** | `tests/test_workflow.py`, `scripts/run_demo.py` |
| Retry limit, no-test-weakening, cancel, gates, persistence, secrets redaction | **Verified, executed** | `tests/test_workflow.py`, `test_services.py`, `test_security.py` |
| Local sandbox (timeout, memory cap, output cap, scrubbed env) | **Verified, executed** — but **NOT a security boundary** | `tests/test_sandbox.py` |
| Docker sandbox | Implemented; command policy unit-tested; **never executed** (no Docker in build env) | `test_security.py::TestSandboxCommandPolicy` |
| Gemini LLM provider | Implemented on `google-genai`; **not live-tested**; LLM-mode branches tested with a stub test double | `test_integrations.py`, `test_workflow.py::TestLLMModePath` |
| GitHub integration | Implemented on the REST API; tested against a fake transport; **no live call made** | `test_integrations.py` |
| Hugging Face datasets | Metadata lookup only; **not live-tested**; bundled Iris is validated for real | `test_integrations.py` |
| FastAPI + Pydantic HTTP layer | Written, `py_compile`d; **API tests exist but were skipped** (FastAPI not installable offline) — run them yourself | `tests/test_api.py` |
| Frontend (vanilla JS SPA) | JS syntax-checked only; **never opened in a browser** | `frontend/index.html` |
| LangGraph | **Not used** (offline build). Explicit state machine in `graph.py`; nodes are plain functions over a typed state | `graph.py` |
| PostgreSQL | **Not implemented**. SQLite with versioned migrations | `storage.py` |

Last test run (~45 s): `Ran 112 tests — OK (skipped=7)`: 105 passed, 0 failed, 5 API tests skipped (no FastAPI in the build
environment), 2 live-credential tests skipped. Skipped tests are never counted as passed.

**Close the remaining gaps on your machine in one command:** `python scripts/verify_setup.py` runs the full suite including the
API tests, boots the FastAPI app and drives the demo over HTTP, probes Docker isolation (network blocked / read-only rootfs /
no secrets), and makes live Gemini, GitHub and Hugging Face calls when configured. Then follow `docs/VERIFY.md` (3-minute
browser checklist). Paste the output into your submission as evidence.

## The demo paper — and why it is a fixture

`sample_papers/fisher_lda_iris_fixture.pdf` is a **synthetic test fixture written for this project, not a published
paper.** It describes Fisher's (1936) linear discriminant method on the public Iris data. The "reported" accuracy
(0.98) is a **fixture reference value, not a published benchmark** (Fisher did not report accuracy). It also contains a
deliberate prompt-injection sentence to exercise the scanner. For a real-paper demo, replace it with a genuine paper
whose method/dataset you have verified (LLM key required for anything other than this template).

Real result from the shipped demo (45 test samples, seed 42): accuracy **0.9556** vs fixture reference 0.98 →
2.5 % relative difference → **Partially reproduced** (tolerances: ≤2 % reproduced, ≤10 % partial). Dataset and split
(Iris, 70/30) match, so the comparison is valid; the exact original split is unknown and is flagged as a warning.

## Quick start

```bash
cd paper2product/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env          # optional: add GEMINI_API_KEY / GITHUB_TOKEN
uvicorn app.main:app --reload       # open http://localhost:8000
```

No API key? It runs in **DEMO MODE** (rule-based extraction + deterministic template; the UI/report/manifest say so).

Headless demo (no server needed): `python scripts/run_demo.py` → writes `demo_output/REPORT.md`, manifest, `project.zip`.

Docker: `docker compose up --build` (see isolation note below).

### Environment variables
`GEMINI_API_KEY`, `P2P_LLM_PROVIDER` (auto|gemini|demo), `P2P_LLM_MODEL`, `GITHUB_TOKEN`, `HF_TOKEN`,
`P2P_SANDBOX` (auto|docker|local), `P2P_SANDBOX_IMAGE`, `P2P_DATA_DIR`, `P2P_API_KEY`, `P2P_MAX_UPLOAD_MB`,
`P2P_RATE_LIMIT_PER_MIN`. See `.env.example`.

### Configuring integrations
* **LLM:** `export GEMINI_API_KEY=...`; UI chip changes from "DEMO MODE" to the model name. Without it you get a clear message, never a fake call.
* **GitHub:** `export GITHUB_TOKEN=...` (fine-grained PAT; `Contents: write`, `Administration: write` only if you want export). Read-only inspect works without a token at low rate limits. Export creates a **new private** repo only after you tick the confirmation box; every request is audit-logged.
* **Datasets:** `sklearn:iris` (bundled), `hf:<id>` (needs `huggingface_hub`; metadata/licence only), `upload:<csv>` (≤50 MB, schema-checked).

### Secure sandbox mode
With Docker available (`P2P_SANDBOX=auto|docker`) runs use `--network none --read-only --cap-drop ALL
--security-opt no-new-privileges --pids-limit --memory --cpus --user 65534`, no Docker socket, no secrets.
Without Docker it falls back to a **local dev sandbox that is not secure** (rlimits + scrubbed env only) and every
result says so. `docker-compose.yml` uses the local sandbox on purpose: mounting the host Docker socket into the API
container would be worse. For real isolation run `uvicorn` on a Docker-equipped host.

## Run the tests
```bash
cd backend
python -m unittest discover -s tests -t .        # works with stdlib only (~45 s); `make test-fast` for a ~5 s subset
pytest                                            # same suite (+ API tests when FastAPI is installed)
P2P_LIVE_TESTS=1 GEMINI_API_KEY=... GITHUB_TOKEN=... pytest tests/test_integrations.py   # optional live tests
```

## Architecture (see `docs/ARCHITECTURE.md`)
`graph.py` state machine: `created → ingesting → awaiting_spec_approval → planning → awaiting_plan_approval →
resolving_data → generating → testing ⇄ repairing → executing → evaluating → reporting → completed`
(+ `blocked | failed | cancelled`). Invalid transitions raise. Two human gates; execution is impossible before both.
Deterministic code does validation, metrics, permissions and status; the LLM (when configured) only interprets
(spec extraction, planning, codegen, repair proposals) and every output is schema-validated.

## Security model
Untrusted: PDFs, repos, dataset metadata, generated code, LLM output. Controls: PDF magic/size/page limits;
injection-pattern scan + `<untrusted_document>` wrapping; path allowlist (no traversal/absolute/hidden/odd
extensions); AST policy scan of generated code; only `python -m unittest` / `python <script>.py` can execute;
patches can never edit tests or add forbidden code; API errors never leak stack traces; secrets redacted in logs,
reports, manifests; rate limiting; optional API key; no GitHub write without explicit confirmation.

## Known limitations & remaining risk
* Local sandbox is **not** secure; Docker mode is untested here. Don't run untrusted papers without Docker mode.
* Without an LLM only the Fisher-LDA/Iris template runs. Generic paper→code needs a model and is **unproven** — LLM code quality varies and results may not match papers.
* Heuristic extraction needs conventional section headings; use the review gate to correct it.
* Cancellation is checked between steps; a running sandbox call finishes or times out first.
* In-memory job queue and rate limiter (single process). Restart marks interrupted runs `failed`.
* Static AST scanning is defence-in-depth, easily bypassed; it is not the security boundary.
* Passing tests ≠ reproduction. Statuses come only from measured-vs-reported comparison with dataset/split checks.
* Not production-ready: no auth beyond a shared key, no multi-tenant isolation, no PostgreSQL, no virus scanning, no load testing.
