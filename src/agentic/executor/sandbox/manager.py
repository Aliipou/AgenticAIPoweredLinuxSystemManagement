"""Docker sandbox manager — executes commands inside scope-constrained containers.

Each call spawns a fresh container (clean slate — no state carried between actions),
runs the command, captures output, and removes the container on exit.

WHAT THIS PROVIDES:
  - Minimal syscall surface per ActionScope via seccomp whitelisting
  - Dropped Linux capabilities (--cap-drop ALL)
  - No silent fallback to host execution if Docker is unavailable

WHAT THIS DOES NOT PROVIDE:
  - Full filesystem isolation for PACKAGE/SERVICE scopes (those actions must
    reach the host to have effect — see SEMANTICS.md)
  - Kernel-level isolation (shares host kernel — not a microVM)
  - Protection against a malicious operator who controls the Pipeline

Docker must be available at runtime. SandboxUnavailableError is raised on failure
to contact the Docker daemon — never silently ignored.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass

from agentic.executor.sandbox.seccomp_profiles import seccomp_json
from agentic.models.action import ActionResult, ActionScope

# Scopes that need access to the host network (apt, systemctl/D-Bus)
_NETWORK_HOST_SCOPES: frozenset[ActionScope] = frozenset({
    ActionScope.PACKAGE,
    ActionScope.SERVICE,
})

# Scope-specific extra docker flags applied at runtime
_SCOPE_FLAGS: dict[ActionScope, list[str]] = {
    # Process operations need the host PID namespace to signal host processes
    ActionScope.PROCESS: ["--pid=host"],
    # Memory management needs /proc/sys/vm on the host
    ActionScope.MEMORY: ["-v", "/proc/sys/vm:/proc/sys/vm"],
    # Service management needs the D-Bus socket
    ActionScope.SERVICE: ["-v", "/run/dbus:/run/dbus:ro"],
    # Package management: no extra mounts; runs inside container
    # (installs into container fs — see SEMANTICS.md for limitation)
    ActionScope.PACKAGE: [],
}


class SandboxUnavailableError(Exception):
    """Raised when the Docker daemon is unreachable or docker is not installed."""


@dataclass(frozen=True)
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int


class SandboxManager:
    """Executes a command inside a fresh Docker container constrained to the
    given ActionScope's syscall whitelist.

    Args:
        image: Docker image to use. Must be available locally or pullable.
        timeout: Hard timeout in seconds per container run.
    """

    def __init__(self, image: str = "ubuntu:22.04", timeout: int = 30) -> None:
        self._image = image
        self._timeout = timeout

    def is_available(self) -> bool:
        """Return True if the Docker daemon is reachable."""
        if shutil.which("docker") is None:
            return False
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    async def run(
        self,
        command: str,
        scope: ActionScope,
        action_id: str = "",
    ) -> ActionResult:
        """Run a shell command inside a fresh, scope-constrained container.

        Raises:
            SandboxUnavailableError: if Docker daemon is unreachable.
        """
        if not self.is_available():
            raise SandboxUnavailableError(
                "Docker daemon is unavailable. Cannot execute in sandbox."
            )

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._run_sync, command, scope
        )

        if result.success:
            return ActionResult(
                action_id=action_id,
                success=True,
                output=result.stdout,
            )
        return ActionResult(
            action_id=action_id,
            success=False,
            output=result.stdout,
            error=result.stderr or f"Container exited with code {result.exit_code}",
        )

    def _run_sync(self, command: str, scope: ActionScope) -> SandboxResult:
        """Build and execute the docker run command synchronously."""
        profile = seccomp_json(scope)

        flags: list[str] = [
            "docker", "run",
            "--rm",
            "--security-opt", f"seccomp={profile}",
            "--cap-drop", "ALL",
        ]

        if scope in _NETWORK_HOST_SCOPES:
            flags += ["--network", "host"]
        else:
            flags += ["--network", "none"]

        flags += _SCOPE_FLAGS.get(scope, [])
        flags += [self._image, "sh", "-c", command]

        try:
            proc = subprocess.run(
                flags,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Command exceeded {self._timeout}s timeout",
                exit_code=-1,
            )
        except OSError as exc:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(exc),
                exit_code=-1,
            )
