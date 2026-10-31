# Verification checklist (run before submitting)

1. `pip install -r backend/requirements.txt`
2. `python scripts/verify_setup.py` — covers: full test suite (incl. API tests), FastAPI boot + full demo flow over HTTP,
   Docker isolation probe, live Gemini call + LLM extraction, live GitHub read, Hugging Face metadata.
   Paste its output into your submission as evidence. SKIP means "not configured", not "passed".
3. **Browser check (manual, 3 min)** at http://localhost:8000:
   - [ ] Header chip shows DEMO MODE (or the Gemini model name if a key is set)
   - [ ] Dashboard → "Start demo (with injected fault)" opens the workspace
   - [ ] Spec gate appears with the red security banner; edit a field; approve
   - [ ] Plan gate: Approve is rejected until the assumptions checkbox is ticked
   - [ ] Live log shows tests failing → diagnose → repair → tests passing → evaluate
   - [ ] Code tab: file tree opens files; "Repair diffs" shows the one-line diff
   - [ ] Experiment tab: bars, confusion matrix, "Reported vs reproduced" with warnings
   - [ ] Report tab: report renders; Download report / manifest / ZIP all work
   - [ ] Stop/cancel button works during a run; upload of a non-PDF shows an error toast
4. Real-paper run (needs `GEMINI_API_KEY`, ideally Docker): upload a genuine small ML paper, review the spec gate
   carefully, run, and read the report's status. Expect to correct the spec by hand — that is the intended workflow.
5. `P2P_SANDBOX=docker uvicorn app.main:app` → UI chip must say `sandbox: docker`.
