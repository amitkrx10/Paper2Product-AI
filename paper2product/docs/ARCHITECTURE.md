# Architecture

```
frontend (SPA) ──HTTP/SSE-ish polling──► FastAPI (main.py: glue, request IDs, error handlers, API key)
                                            │
                                      services.py (validation, rate limit, job runner, audit, GitHub export)
                                            │
                                      graph.py Engine  ── typed WorkflowState persisted in SQLite (storage.py)
   ┌──────────┬───────────┬────────────┬───────────┬──────────────┬──────────────┬────────────┐
 ingest.py  planner.py  datasets.py  codegen.py  sandbox.py     repair.py    evaluation.py → report.py
 (Stage A)  (Stage B)   (Stage C)    (Stage D)   (Stage E)      (Stage F)    (Stage G)       (Stage H)
```
LLM provider abstraction: `llm/provider.py` (Gemini via `google-genai`; `None` = demo mode).

| Node | LLM? | Deterministic responsibilities |
|---|---|---|
| Research Parser | optional (schema-validated) | PDF limits, page refs, injection scan, heuristic extraction |
| Planner | optional | schema + dependency validation, assumptions list, feasibility |
| Resource Resolver | no | provenance, hash, licence, size/schema validation |
| Code Generator | optional | path/size allowlist, AST policy scan, must include tests |
| Test & Execution | no | sandbox invocation allowlist, limits, log capture |
| Debug & Repair | optional (proposal only) | traceback diagnosis, rule patch, patch policy, retry budget |
| Evaluator | no | metric maths from predictions, comparison, status |
| Reporter | no | report, manifest (sha256 of every file), ZIP |

Repair loop: `tests → (fail) diagnose → patch(validate) → tests → (pass) experiment → (fail) diagnose → ...`
bounded by `max_repairs` (default 3), wall-clock and token budgets. Tests are never editable by repairs.
