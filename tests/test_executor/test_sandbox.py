"""Tests for sandbox seccomp profiles, SandboxManager, and ActionExecutor sandbox path."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from agentic.executor.action_executor import ActionExecutor, _ACTION_SCOPES
from agentic.executor.sandbox.manager import SandboxManager, SandboxResult, SandboxUnavailableError
from agentic.executor.sandbox.seccomp_profiles import (
    _BASELINE,
    _SCOPE_EXTRA,
    build_seccomp_profile,
    seccomp_json,
)
from agentic.models.action import ActionCandidate, ActionResult, ActionScope, ActionType


# ---------------------------------------------------------------------------
# Seccomp profile tests
# ---------------------------------------------------------------------------

class TestSeccompProfiles:
    def test_build_returns_dict_with_required_keys(self):
        profile = build_seccomp_profile(ActionScope.PROCESS)
        assert "defaultAction" in profile
        assert "architectures" in profile
        assert "syscalls" in profile

    def test_default_action_is_errno(self):
        for scope in ActionScope:
            profile = build_seccomp_profile(scope)
            assert profile["defaultAction"] == "SCMP_ACT_ERRNO"

    def test_syscalls_entry_is_allow(self):
        profile = build_seccomp_profile(ActionScope.PROCESS)
        assert len(profile["syscalls"]) == 1
        assert profile["syscalls"][0]["action"] == "SCMP_ACT_ALLOW"

    def test_baseline_syscalls_present_in_every_scope(self):
        for scope in ActionScope:
            profile = build_seccomp_profile(scope)
            names = profile["syscalls"][0]["names"]
            for syscall in _BASELINE:
                assert syscall in names, f"{syscall} missing from {scope} profile"

    def test_process_scope_has_kill(self):
        profile = build_seccomp_profile(ActionScope.PROCESS)
        names = profile["syscalls"][0]["names"]
        assert "kill" in names
        assert "tgkill" in names

    def test_process_scope_has_setpriority(self):
        names = build_seccomp_profile(ActionScope.PROCESS)["syscalls"][0]["names"]
        assert "setpriority" in names

    def test_package_scope_has_execve(self):
        names = build_seccomp_profile(ActionScope.PACKAGE)["syscalls"][0]["names"]
        assert "execve" in names

    def test_package_scope_has_network_syscalls(self):
        names = build_seccomp_profile(ActionScope.PACKAGE)["syscalls"][0]["names"]
        assert "socket" in names
        assert "connect" in names

    def test_memory_scope_no_execve(self):
        names = build_seccomp_profile(ActionScope.MEMORY)["syscalls"][0]["names"]
        assert "execve" not in names

    def test_service_scope_has_execve_and_socket(self):
        names = build_seccomp_profile(ActionScope.SERVICE)["syscalls"][0]["names"]
        assert "execve" in names
        assert "socket" in names
        assert "connect" in names

    def test_no_duplicate_syscalls(self):
        for scope in ActionScope:
            names = build_seccomp_profile(scope)["syscalls"][0]["names"]
            assert len(names) == len(set(names)), f"Duplicates in {scope} profile"

    def test_scope_extra_keys_are_valid_scopes(self):
        for scope in _SCOPE_EXTRA:
            assert isinstance(scope, ActionScope)

    def test_seccomp_json_is_valid_json(self):
        for scope in ActionScope:
            raw = seccomp_json(scope)
            parsed = json.loads(raw)
            assert parsed["defaultAction"] == "SCMP_ACT_ERRNO"

    def test_seccomp_json_no_spaces(self):
        raw = seccomp_json(ActionScope.PROCESS)
        assert " " not in raw


# ---------------------------------------------------------------------------
# SandboxManager availability check
# ---------------------------------------------------------------------------

class TestSandboxManagerAvailability:
    def test_unavailable_when_docker_not_on_path(self):
        manager = SandboxManager()
        with patch("shutil.which", return_value=None):
            assert manager.is_available() is False

    def test_unavailable_when_docker_info_fails(self):
        manager = SandboxManager()
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert manager.is_available() is False

    def test_available_when_docker_info_succeeds(self):
        manager = SandboxManager()
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert manager.is_available() is True

    def test_unavailable_on_timeout(self):
        manager = SandboxManager()
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 5)):
            assert manager.is_available() is False

    def test_unavailable_on_os_error(self):
        manager = SandboxManager()
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=OSError("no such file")):
            assert manager.is_available() is False


# ---------------------------------------------------------------------------
# SandboxManager._run_sync internals
# ---------------------------------------------------------------------------

class TestSandboxManagerRunSync:
    def _manager(self) -> SandboxManager:
        return SandboxManager(image="ubuntu:22.04", timeout=30)

    def test_run_sync_success(self):
        manager = self._manager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
            result = manager._run_sync("echo ok", ActionScope.PROCESS)
        assert result.success is True
        assert result.stdout == "ok\n"
        assert result.exit_code == 0

    def test_run_sync_failure(self):
        manager = self._manager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            result = manager._run_sync("false", ActionScope.PROCESS)
        assert result.success is False
        assert result.stderr == "error"

    def test_run_sync_timeout(self):
        manager = self._manager()
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 30)):
            result = manager._run_sync("sleep 999", ActionScope.PROCESS)
        assert result.success is False
        assert "timeout" in result.stderr.lower()
        assert result.exit_code == -1

    def test_run_sync_os_error(self):
        manager = self._manager()
        with patch("subprocess.run", side_effect=OSError("docker not found")):
            result = manager._run_sync("echo x", ActionScope.PROCESS)
        assert result.success is False
        assert result.exit_code == -1

    def test_run_sync_includes_rm_flag(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("echo x", ActionScope.PROCESS)
        assert "--rm" in captured[0]

    def test_run_sync_includes_cap_drop_all(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("echo x", ActionScope.MEMORY)
        assert "--cap-drop" in captured[0]
        idx = captured[0].index("--cap-drop")
        assert captured[0][idx + 1] == "ALL"

    def test_process_scope_uses_no_network(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("echo x", ActionScope.PROCESS)
        assert "--network" in captured[0]
        idx = captured[0].index("--network")
        assert captured[0][idx + 1] == "none"

    def test_package_scope_uses_host_network(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("apt install vim", ActionScope.PACKAGE)
        assert "--network" in captured[0]
        idx = captured[0].index("--network")
        assert captured[0][idx + 1] == "host"

    def test_service_scope_uses_host_network(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("systemctl restart nginx", ActionScope.SERVICE)
        idx = captured[0].index("--network")
        assert captured[0][idx + 1] == "host"

    def test_memory_scope_uses_no_network(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("sync", ActionScope.MEMORY)
        idx = captured[0].index("--network")
        assert captured[0][idx + 1] == "none"

    def test_process_scope_has_pid_host_flag(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("kill -9 123", ActionScope.PROCESS)
        assert "--pid=host" in captured[0]

    def test_memory_scope_mounts_proc_sys(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("echo 3 > /proc/sys/vm/drop_caches", ActionScope.MEMORY)
        assert "-v" in captured[0]
        idx = captured[0].index("-v")
        assert "/proc/sys/vm" in captured[0][idx + 1]

    def test_service_scope_mounts_dbus(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("systemctl stop nginx", ActionScope.SERVICE)
        cmd_str = " ".join(captured[0])
        assert "/run/dbus" in cmd_str

    def test_seccomp_flag_present(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("echo x", ActionScope.PROCESS)
        assert "--security-opt" in captured[0]
        idx = captured[0].index("--security-opt")
        assert captured[0][idx + 1].startswith("seccomp=")

    def test_seccomp_value_is_valid_json(self):
        manager = self._manager()
        captured: list[list[str]] = []
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (captured.append(cmd), MagicMock(returncode=0, stdout="", stderr=""))[1]
            manager._run_sync("echo x", ActionScope.PACKAGE)
        idx = captured[0].index("--security-opt")
        seccomp_arg = captured[0][idx + 1]
        raw_json = seccomp_arg[len("seccomp="):]
        parsed = json.loads(raw_json)
        assert parsed["defaultAction"] == "SCMP_ACT_ERRNO"


# ---------------------------------------------------------------------------
# SandboxManager.run() async path
# ---------------------------------------------------------------------------

class TestSandboxManagerRunAsync:
    @pytest.mark.asyncio
    async def test_run_raises_when_unavailable(self):
        manager = SandboxManager()
        with patch.object(manager, "is_available", return_value=False):
            with pytest.raises(SandboxUnavailableError):
                await manager.run("echo x", ActionScope.PROCESS, action_id="a1")

    @pytest.mark.asyncio
    async def test_run_returns_success_result(self):
        manager = SandboxManager()
        with patch.object(manager, "is_available", return_value=True), \
             patch.object(manager, "_run_sync", return_value=SandboxResult(
                 success=True, stdout="output", stderr="", exit_code=0
             )):
            result = await manager.run("echo x", ActionScope.PROCESS, action_id="a1")
        assert result.success is True
        assert result.output == "output"
        assert result.action_id == "a1"

    @pytest.mark.asyncio
    async def test_run_returns_failure_result(self):
        manager = SandboxManager()
        with patch.object(manager, "is_available", return_value=True), \
             patch.object(manager, "_run_sync", return_value=SandboxResult(
                 success=False, stdout="", stderr="container error", exit_code=1
             )):
            result = await manager.run("bad", ActionScope.PROCESS, action_id="b1")
        assert result.success is False
        assert result.error == "container error"

    @pytest.mark.asyncio
    async def test_run_failure_uses_exit_code_when_no_stderr(self):
        manager = SandboxManager()
        with patch.object(manager, "is_available", return_value=True), \
             patch.object(manager, "_run_sync", return_value=SandboxResult(
                 success=False, stdout="", stderr="", exit_code=137
             )):
            result = await manager.run("oom", ActionScope.PROCESS, action_id="c1")
        assert result.success is False
        assert "137" in result.error


# ---------------------------------------------------------------------------
# ActionExecutor sandbox path
# ---------------------------------------------------------------------------

class TestActionExecutorSandbox:
    def _action(self, action_type: ActionType = ActionType.KILL_PROCESS) -> ActionCandidate:
        return ActionCandidate(
            action_type=action_type,
            description="test",
            command="kill -9 123",
            target="chrome",
        )

    @pytest.mark.asyncio
    async def test_without_sandbox_uses_runner(self):
        executor = ActionExecutor(sandbox=None)
        action = self._action(ActionType.SUSPEND_PROCESS)
        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=ActionResult(action_id=action.id, success=True))
        with patch.object(executor, "_get_runner", return_value=mock_runner):
            result = await executor.execute(action, dry_run=False)
        assert result.success is True
        mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_sandbox_bypasses_runner(self):
        mock_sandbox = AsyncMock(spec=SandboxManager)
        mock_sandbox.run = AsyncMock(return_value=ActionResult(action_id="x", success=True, output="sandboxed"))
        executor = ActionExecutor(sandbox=mock_sandbox)
        action = self._action(ActionType.KILL_PROCESS)

        with patch.object(executor, "_get_runner") as mock_get_runner:
            result = await executor.execute(action, dry_run=False)
            mock_get_runner.assert_not_called()

        assert result.output == "sandboxed"
        mock_sandbox.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_sandbox_receives_correct_scope(self):
        mock_sandbox = AsyncMock(spec=SandboxManager)
        mock_sandbox.run = AsyncMock(return_value=ActionResult(action_id="x", success=True))
        executor = ActionExecutor(sandbox=mock_sandbox)

        action = self._action(ActionType.APT_INSTALL)
        await executor.execute(action)

        _, kwargs = mock_sandbox.run.call_args
        assert kwargs.get("scope") == ActionScope.PACKAGE or mock_sandbox.run.call_args[0][1] == ActionScope.PACKAGE

    @pytest.mark.asyncio
    async def test_dry_run_bypasses_sandbox(self):
        mock_sandbox = AsyncMock(spec=SandboxManager)
        executor = ActionExecutor(sandbox=mock_sandbox)
        action = self._action(ActionType.SUSPEND_PROCESS)

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock(return_value=ActionResult(action_id=action.id, success=True, output="DRY RUN"))
        with patch.object(executor, "_get_runner", return_value=mock_runner):
            result = await executor.execute(action, dry_run=True)

        mock_sandbox.run.assert_not_called()
        assert result.success is True

    def test_action_scope_map_covers_all_action_types(self):
        for at in ActionType:
            assert at in _ACTION_SCOPES, f"{at} missing from _ACTION_SCOPES"

    def test_action_scope_map_values_are_action_scopes(self):
        for at, scope in _ACTION_SCOPES.items():
            assert isinstance(scope, ActionScope)
