import os, tempfile, time, unittest
from app.sandbox import LocalSandbox, SandboxResult


def run(script, **kw):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "t.py"), "w") as f:
        f.write(script)
    args = dict(timeout_s=5, memory_mb=512, max_output_kb=64); args.update(kw)
    return LocalSandbox().run_python(d, ["t.py"], **args)


class TestLocalSandbox(unittest.TestCase):
    def test_captures_stdout_stderr_exit(self):
        r = run("import sys; print('hi'); print('err', file=sys.stderr); sys.exit(3)")
        self.assertEqual((r.exit_code, r.stdout.strip(), r.stderr.strip()), (3, "hi", "err"))
        self.assertFalse(r.ok)

    def test_timeout_kills_runaway(self):
        t = time.time()
        r = run("while True: pass", timeout_s=1)
        self.assertTrue(r.timed_out); self.assertLess(time.time() - t, 6); self.assertFalse(r.ok)

    def test_memory_limit(self):
        r = run("x = bytearray(900*1024*1024)\nprint('allocated')", memory_mb=200)
        self.assertNotEqual(r.exit_code, 0); self.assertNotIn("allocated", r.stdout)

    def test_output_truncation(self):
        r = run("print('x' * 500000)", max_output_kb=1)
        self.assertTrue(r.truncated); self.assertLess(len(r.stdout), 2000)

    def test_secrets_not_inherited(self):
        os.environ["P2P_TEST_SECRET"] = "topsecret"
        try:
            r = run("import os; print(os.environ.get('P2P_TEST_SECRET')); print(sorted(os.environ))")
            self.assertIn("None", r.stdout); self.assertNotIn("topsecret", r.stdout)
        finally:
            del os.environ["P2P_TEST_SECRET"]

    def test_local_mode_is_labelled_not_secure(self):
        r = run("print(1)")
        self.assertFalse(r.isolated); self.assertIn("NOT SECURE", r.note); self.assertEqual(r.mode, "local")

    def test_no_stdin(self):
        r = run("import sys; print(repr(sys.stdin.read()))")
        self.assertIn("''", r.stdout)
