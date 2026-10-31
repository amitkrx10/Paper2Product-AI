# Live demo script (≈4 min) and pitch

**Before:** `uvicorn app.main:app` → open http://localhost:8000. (Say out loud: "No API key, so this is DEMO MODE —
deterministic template, but every execution and metric you see is real.")

1. Dashboard → **Start demo (with injected fault)**. Point at the DEMO MODE and "sandbox: local (NOT isolated)" chips.
2. **Spec gate:** show page-referenced extraction, the red security banner (prompt injection in the PDF — treated as data), edit a field.
3. Approve → **plan gate:** read the assumption ("exact split unknown"), tick the box, **Approve & run**.
4. **Live activity:** tests fail (`errors=4`) → `diagnose: NameError 'Sw_inverse'` → `repair attempt 1/3` → tests pass → experiment → evaluate.
5. **Code & diffs:** show the one-line diff; note the agent cannot touch `tests/`.
6. **Experiment:** accuracy 0.9556 vs reference 0.98 → *Partially reproduced*, with warnings. "A passing test is not a reproduction."
7. **Report:** download ZIP + manifest; (optional) show the GitHub export requires an explicit confirmation.
8. Mention: set `GEMINI_API_KEY` to unlock arbitrary papers; run with Docker for real isolation.

**Pitch (30 s):** "Reproducing a paper costs days. Paper2Product is a tool-using agent that reads the paper, asks you to approve
its understanding, writes a tested project, runs it in a sandbox, fixes its own bugs within a budget, measures the
metrics itself, and tells you honestly whether the paper reproduced. It never fakes success: every claim in the report
links to an execution it actually performed."

**Three differentiators:** (1) Evidence-first honesty — independent metric recomputation, valid-comparison checks, four-level status, DEMO/sandbox labelling in every artifact. (2) Safe, bounded autonomy — human approval gates, repair budget, repairs can't weaken tests, AST + sandbox + path controls. (3) Real developer-tool integrations — LLM, GitHub (confirm-gated writes + audit), dataset provenance, Docker execution — with a reproducibility manifest (sha256 of every file).
