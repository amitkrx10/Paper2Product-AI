.PHONY: install run test demo test-fast verify
install: ; pip install -r backend/requirements.txt
run: ; cd backend && uvicorn app.main:app --reload
test: ; cd backend && python -m unittest discover -s tests -t .
demo: ; python scripts/run_demo.py

test-fast: ; cd backend && python -m unittest tests.test_security tests.test_evaluation tests.test_ingest_schemas tests.test_integrations tests.test_sandbox
verify: ; python scripts/verify_setup.py
