"""Docker-based Python sandbox for safe code execution.

Executes untrusted LLM-generated code in an isolated container with:
- No network access
- Memory and CPU limits
- Configurable timeout
- Captured stdout/stderr and exit code

Used for:
1. Quality gate Stage A — running reference solutions against test suites
2. Dual-blind sandbox execution — running model outputs against Test-A and Test-B
"""

from __future__ import annotations

import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import docker
from docker.errors import ContainerError, ImageNotFound

_DEFAULT_IMAGE = "python:3.11-slim"
_DEFAULT_TIMEOUT_S = 30
_DEFAULT_MEM_LIMIT = "256m"


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox:
    """Docker-based sandbox for executing Python code + tests.

    Usage:
        sandbox = Sandbox()

        # Run code with a test suite
        result = sandbox.run(
            code="def add(a, b): return a + b",
            test_code="assert add(1, 2) == 3",
        )
        print(result.passed)  # True

        # Run a complete script
        result = sandbox.run_script("print('hello')")
        print(result.stdout)  # "hello\\n"
    """

    def __init__(
        self,
        image: str = _DEFAULT_IMAGE,
        timeout_s: int = _DEFAULT_TIMEOUT_S,
        mem_limit: str = _DEFAULT_MEM_LIMIT,
    ):
        self.image = image
        self.timeout_s = timeout_s
        self.mem_limit = mem_limit
        self._client = self._connect_docker()
        self._ensure_image()

    @staticmethod
    def _connect_docker() -> docker.DockerClient:
        """Connect to Docker, trying multiple socket locations."""
        # Docker Desktop on macOS uses a user-scoped socket
        socket_paths = [
            Path.home() / ".docker" / "run" / "docker.sock",
            Path("/var/run/docker.sock"),
        ]
        for sock in socket_paths:
            if sock.exists():
                return docker.DockerClient(base_url=f"unix://{sock}")
        # Fall back to default env-based detection
        return docker.from_env()

    def _ensure_image(self) -> None:
        try:
            self._client.images.get(self.image)
        except ImageNotFound:
            print(f"Pulling Docker image {self.image}...")
            self._client.images.pull(self.image)

    def run(
        self,
        code: str,
        test_code: str,
        timeout_s: Optional[int] = None,
    ) -> SandboxResult:
        """Execute `code` then run `test_code` assertions against it.

        The code and test_code are concatenated into a single script:
            <code>
            <test_code>

        If all assertions pass, exit_code is 0.
        """
        script = code.rstrip("\n") + "\n\n" + test_code.rstrip("\n") + "\n"
        return self.run_script(script, timeout_s=timeout_s)

    def run_script(
        self,
        script: str,
        timeout_s: Optional[int] = None,
    ) -> SandboxResult:
        """Execute an arbitrary Python script in the sandbox."""
        timeout = timeout_s or self.timeout_s

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "run.py"
            script_path.write_text(script, encoding="utf-8")

            # Wrapper that enforces the timeout from inside the container
            # and captures the exit code properly.
            wrapper = textwrap.dedent(f"""\
                import subprocess, sys
                try:
                    result = subprocess.run(
                        [sys.executable, "/sandbox/run.py"],
                        capture_output=True, text=True,
                        timeout={timeout},
                    )
                    sys.stdout.write(result.stdout)
                    sys.stderr.write(result.stderr)
                    sys.exit(result.returncode)
                except subprocess.TimeoutExpired:
                    sys.stderr.write("SANDBOX_TIMEOUT")
                    sys.exit(124)
            """)
            wrapper_path = Path(tmpdir) / "wrapper.py"
            wrapper_path.write_text(wrapper, encoding="utf-8")

            try:
                container = self._client.containers.run(
                    image=self.image,
                    command=["python", "/sandbox/wrapper.py"],
                    volumes={tmpdir: {"bind": "/sandbox", "mode": "ro"}},
                    network_mode="none",
                    mem_limit=self.mem_limit,
                    # pids_limit prevents fork bombs
                    pids_limit=64,
                    detach=True,
                )

                # Wait with an outer timeout (slightly longer for container overhead)
                wait_result = container.wait(timeout=timeout + 10)
                exit_code = wait_result.get("StatusCode", 1)

                stdout = container.logs(stdout=True, stderr=False).decode(
                    "utf-8", errors="replace"
                )
                stderr = container.logs(stdout=False, stderr=True).decode(
                    "utf-8", errors="replace"
                )
                timed_out = exit_code == 124 and "SANDBOX_TIMEOUT" in stderr

            except Exception as e:
                # Container hung or other Docker error
                stdout = ""
                stderr = str(e)
                exit_code = 1
                timed_out = True
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
        )

    def run_dual_blind(
        self,
        code: str,
        test_a: str,
        test_b: str,
        timeout_s: Optional[int] = None,
    ) -> tuple[SandboxResult, SandboxResult]:
        """Run code against both test suites (dual-blind execution).

        Returns (result_a, result_b) — used to compute pass@k(A) and pass@k(B).
        """
        result_a = self.run(code, test_a, timeout_s=timeout_s)
        result_b = self.run(code, test_b, timeout_s=timeout_s)
        return result_a, result_b

    def validate_quality_gate_a(
        self,
        ref_solution_a: str,
        ref_solution_b: str,
        test_a: str,
        test_b: str,
        timeout_s: Optional[int] = None,
    ) -> bool:
        """Quality Gate Stage A: mechanical sandbox check.

        Verifies strict exclusivity:
        - Reference A passes Test-A and fails Test-B
        - Reference B passes Test-B and fails Test-A
        """
        ra_ta = self.run(ref_solution_a, test_a, timeout_s=timeout_s)
        ra_tb = self.run(ref_solution_a, test_b, timeout_s=timeout_s)
        rb_ta = self.run(ref_solution_b, test_a, timeout_s=timeout_s)
        rb_tb = self.run(ref_solution_b, test_b, timeout_s=timeout_s)

        return (
            ra_ta.passed
            and not ra_tb.passed
            and not rb_ta.passed
            and rb_tb.passed
        )
