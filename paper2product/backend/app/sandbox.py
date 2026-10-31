"""Stage E - controlled execution.

DockerSandbox  : real isolation (no network, read-only root, dropped capabilities, resource limits,
                 non-root user, no docker socket, no secrets). Used automatically when Docker works.
LocalSandbox   : development fallback. Scrubbed env + rlimits + timeout + process-group kill.
                 It is NOT a security boundary (no network or filesystem isolation) and is labelled as such
                 in every result, report and manifest.
"""
from __future__ import annotations
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict

from .security import safe_relpath, SecurityError


@dataclass
class SandboxResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    truncated: bool = False
    mode: str = "local"
    isolated: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


import re as _re
_ARG_RX = _re.compile(r"^[A-Za-z0-9_./-]{1,100}$")


def _check_arg(a: str) -> None:
    if not _ARG_RX.match(a) or ".." in a or a.startswith("/"):
        raise SecurityError(f"unsafe argument: {a!r}")


def validate_python_args(args: list[str]) -> list[str]:
    """Only two invocation shapes are allowed: `-m unittest ...` or `<relative script>.py ...`."""
    if not args:
        raise SecurityError("empty command")
    if args[0] == "-m":
        if len(args) < 2 or args[1] != "unittest":
            raise SecurityError("only 'python -m unittest' is allowed")
        for a in args[2:]:
            if a.startswith("-") and a in ("-s", "-t", "-v", "-p"):
                continue
            _check_arg(a)
        return list(args)
    script = safe_relpath(args[0])
    if not script.endswith(".py"):
        raise SecurityError("only .py scripts may be executed")
    for a in args[1:]:
        _check_arg(a)
    return [script] + list(args[1:])


def _clip(b: bytes, limit: int) -> tuple[str, bool]:
    t = len(b) > limit
    return b[:limit].decode("utf-8", errors="replace") + ("\n[output truncated]" if t else ""), t


class LocalSandbox:
    mode = "local"
    isolated = False
    description = ("LOCAL DEVELOPMENT MODE - NOT SECURE: subprocess with resource limits and a scrubbed environment, "
                   "but no network or filesystem isolation. Use Docker for untrusted code.")

    def run_python(self, workdir: str, args: list[str], timeout_s: int, memory_mb: int, max_output_kb: int) -> SandboxResult:
        args = validate_python_args(args)
        argv = [sys.executable, "-E", "-s"] + args
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": workdir, "LANG": "C.UTF-8",
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

        def limits() -> None:
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 5, timeout_s + 5))
            mem = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        tmp = tempfile.mkdtemp(prefix="p2p_out_")
        out_path, err_path = os.path.join(tmp, "out"), os.path.join(tmp, "err")
        start = time.time()
        timed_out = False
        try:
            with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
                proc = subprocess.Popen(argv, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
                                        stdout=fo, stderr=fe, preexec_fn=limits)
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            limit = max_output_kb * 1024
            with open(out_path, "rb") as f:
                out, t1 = _clip(f.read(limit + 1), limit)
            with open(err_path, "rb") as f:
                err, t2 = _clip(f.read(limit + 1), limit)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        code = proc.returncode if not timed_out else -9
        return SandboxResult(argv=["python"] + args, exit_code=code, stdout=out, stderr=err,
                             duration_s=round(time.time() - start, 3), timed_out=timed_out,
                             truncated=t1 or t2, mode="local", isolated=False, note=self.description)


class DockerSandbox:
    mode = "docker"
    isolated = True
    description = "Docker isolation: --network none, read-only rootfs, cap-drop ALL, non-root, resource limits."

    def __init__(self, image: str = "python:3.12-slim"):
        self.image = image

    @staticmethod
    def available() -> bool:
        if not shutil.which("docker"):
            return False
        try:
            return subprocess.run(["docker", "info"], capture_output=True, timeout=8).returncode == 0
        except Exception:
            return False

    def build_command(self, workdir: str, args: list[str], timeout_s: int, memory_mb: int, name: str) -> list[str]:
        args = validate_python_args(args)
        return ["docker", "run", "--rm", "--name", name,
                "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m,noexec",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "128", "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
                "--cpus", "1.0", "--user", "65534:65534",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "HOME=/tmp",
                "-v", f"{os.path.realpath(workdir)}:/work:rw", "-w", "/work",
                self.image, "python"] + args

    def run_python(self, workdir: str, args: list[str], timeout_s: int, memory_mb: int, max_output_kb: int) -> SandboxResult:
        name = f"p2p-{uuid.uuid4().hex[:10]}"
        cmd = self.build_command(workdir, args, timeout_s, memory_mb, name)
        start = time.time()
        timed_out = False
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout_s + 10)
            out, err, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            subprocess.run(["docker", "kill", name], capture_output=True)
            out, err, code = e.stdout or b"", e.stderr or b"", -9
        limit = max_output_kb * 1024
        o, t1 = _clip(out, limit)
        er, t2 = _clip(err, limit)
        return SandboxResult(argv=["python"] + list(args), exit_code=code, stdout=o, stderr=er,
                             duration_s=round(time.time() - start, 3), timed_out=timed_out,
                             truncated=t1 or t2, mode="docker", isolated=True, note=self.description)


def get_sandbox(mode: str = "auto", image: str = "python:3.12-slim"):
    if mode == "local":
        return LocalSandbox()
    if mode == "docker":
        if not DockerSandbox.available():
            raise RuntimeError("P2P_SANDBOX=docker but Docker is not available")
        return DockerSandbox(image)
    return DockerSandbox(image) if DockerSandbox.available() else LocalSandbox()
