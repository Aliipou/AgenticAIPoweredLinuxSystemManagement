"""Brutal tests for action executor and all runners."""

from __future__ import annotations

import signal
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from agentic.exceptions import ExecutionError
from agentic.executor.action_executor import ActionExecutor
from agentic.executor.runners.memory_runner import MemoryRunner
from agentic.executor.runners.package_runner import PackageRunner
from agentic.executor.runners.process_runner import ESSENTIAL_PROCESSES, ProcessRunner, _SIGCONT, _SIGSTOP
from agentic.executor.runners.systemctl_runner import SystemctlRunner
from agentic.models.action import ActionCandidate, ActionType


class TestProcessRunner:
    @pytest.mark.asyncio
    async def test_dry_run(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend firefox",
            target="firefox",
        )
        result = await runner.run(action, dry_run=True)
        assert result.success is True
        assert "DRY RUN" in result.output

    @pytest.mark.asyncio
    async def test_essential_process_rejected(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Kill systemd",
            target="systemd",
        )
        with pytest.raises(ExecutionError, match="essential process"):
            await runner.run(action)

    @pytest.mark.asyncio
    async def test_all_essential_processes_protected(self):
        runner = ProcessRunner()
        for proc_name in ESSENTIAL_PROCESSES:
            action = ActionCandidate(
                action_type=ActionType.KILL_PROCESS,
                description=f"Kill {proc_name}",
                target=proc_name,
            )
            with pytest.raises(ExecutionError, match="essential process"):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_no_matching_processes(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend nonexistent",
            target="nonexistent_process_xyz_12345",
        )
        with patch.object(runner, "_find_pids", return_value=[]):
            result = await runner.run(action)
        assert result.success is True
        assert "No matching" in result.output

    @pytest.mark.asyncio
    async def test_suspend_sends_sigstop(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend test",
            target="testproc",
        )
        mock_proc = MagicMock()
        with (
            patch.object(runner, "_find_pids", return_value=[12345]),
            patch("agentic.executor.runners.process_runner.psutil.Process", return_value=mock_proc),
        ):
            result = await runner.run(action)
        mock_proc.send_signal.assert_called_once_with(_SIGSTOP)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_kill_sends_sigterm(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Kill test",
            target="testproc",
        )
        mock_proc = MagicMock()
        with (
            patch.object(runner, "_find_pids", return_value=[99]),
            patch("agentic.executor.runners.process_runner.psutil.Process", return_value=mock_proc),
        ):
            result = await runner.run(action)
        mock_proc.send_signal.assert_called_once_with(signal.SIGTERM)

    @pytest.mark.asyncio
    async def test_process_access_denied_raises(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Kill protected",
            target="protected",
        )
        mock_proc = MagicMock()
        mock_proc.send_signal.side_effect = psutil.AccessDenied(pid=1)
        with (
            patch.object(runner, "_find_pids", return_value=[1]),
            patch("agentic.executor.runners.process_runner.psutil.Process", return_value=mock_proc),
        ):
            with pytest.raises(ExecutionError, match="Failed to signal"):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_process_no_such_process_raises(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Kill gone",
            target="gone",
        )
        mock_proc = MagicMock()
        mock_proc.send_signal.side_effect = psutil.NoSuchProcess(pid=999)
        with (
            patch.object(runner, "_find_pids", return_value=[999]),
            patch("agentic.executor.runners.process_runner.psutil.Process", return_value=mock_proc),
        ):
            with pytest.raises(ExecutionError):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_rollback_suspend_sends_sigcont(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend test",
            target="testproc",
        )
        mock_proc = MagicMock()
        with (
            patch.object(runner, "_find_pids", return_value=[123]),
            patch("agentic.executor.runners.process_runner.psutil.Process", return_value=mock_proc),
        ):
            result = await runner.rollback(action)
        mock_proc.send_signal.assert_called_with(_SIGCONT)
        assert result.rolled_back is True

    @pytest.mark.asyncio
    async def test_rollback_kill_not_supported(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Kill test",
            target="testproc",
        )
        result = await runner.rollback(action)
        assert result.success is False
        assert "not supported" in result.error

    @pytest.mark.asyncio
    async def test_rollback_suspend_handles_no_such_process(self):
        runner = ProcessRunner()
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend test",
            target="testproc",
        )
        mock_proc = MagicMock()
        mock_proc.send_signal.side_effect = psutil.NoSuchProcess(pid=999)
        with (
            patch.object(runner, "_find_pids", return_value=[999]),
            patch("agentic.executor.runners.process_runner.psutil.Process", return_value=mock_proc),
        ):
            result = await runner.rollback(action)
        assert result.rolled_back is True

    def test_find_pids_matches_by_name(self):
        mock_proc = MagicMock()
        mock_proc.info = {"name": "firefox", "cmdline": ["/usr/bin/firefox"]}
        mock_proc.pid = 42

        with patch("agentic.executor.runners.process_runner.psutil.process_iter", return_value=[mock_proc]):
            pids = ProcessRunner._find_pids("firefox")
        assert 42 in pids

    def test_find_pids_matches_by_cmdline(self):
        mock_proc = MagicMock()
        mock_proc.info = {"name": "python3", "cmdline": ["python3", "myapp.py"]}
        mock_proc.pid = 55

        with patch("agentic.executor.runners.process_runner.psutil.process_iter", return_value=[mock_proc]):
            pids = ProcessRunner._find_pids("myapp")
        assert 55 in pids

    def test_find_pids_handles_access_denied(self):
        mock_proc = MagicMock()
        mock_proc.info = {"name": None, "cmdline": None}

        with patch("agentic.executor.runners.process_runner.psutil.process_iter", return_value=[mock_proc]):
            pids = ProcessRunner._find_pids("test")
        assert pids == []

    def test_find_pids_handles_nosuchprocess_exception(self):
        mock_proc = MagicMock()
        type(mock_proc).info = property(
            lambda self: (_ for _ in ()).throw(psutil.NoSuchProcess(1))
        )
        with patch("agentic.executor.runners.process_runner.psutil.process_iter", return_value=[mock_proc]):
            pids = ProcessRunner._find_pids("test")
        assert pids == []

    def test_find_pids_handles_access_denied_exception(self):
        mock_proc = MagicMock()
        type(mock_proc).info = property(
            lambda self: (_ for _ in ()).throw(psutil.AccessDenied(1))
        )
        with patch("agentic.executor.runners.process_runner.psutil.process_iter", return_value=[mock_proc]):
            pids = ProcessRunner._find_pids("test")
        assert pids == []

    def test_get_signal_mapping(self):
        assert ProcessRunner._get_signal(ActionType.KILL_PROCESS) == signal.SIGTERM
        assert ProcessRunner._get_signal(ActionType.SUSPEND_PROCESS) == _SIGSTOP
        assert ProcessRunner._get_signal(ActionType.RENICE_PROCESS) == signal.SIGTERM

    def test_get_signal_unknown_defaults_to_sigterm(self):
        assert ProcessRunner._get_signal(ActionType.APT_INSTALL) == signal.SIGTERM


class TestPackageRunner:
    @pytest.mark.asyncio
    async def test_dry_run(self):
        runner = PackageRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_INSTALL,
            description="Install vim",
            command="apt install -y vim",
            target="vim",
        )
        result = await runner.run(action, dry_run=True)
        assert result.success is True
        assert "DRY RUN" in result.output

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        runner = PackageRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_INSTALL,
            description="Install vim",
            command="echo 'installed'",
            target="vim",
        )
        result = await runner.run(action)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_failed_command(self):
        runner = PackageRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_INSTALL,
            description="Install bad",
            command="false",
            target="bad",
        )
        with pytest.raises(ExecutionError, match="failed"):
            await runner.run(action)

    @pytest.mark.asyncio
    async def test_os_error(self):
        runner = PackageRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_INSTALL,
            description="Install",
            command="/nonexistent/binary",
            target="pkg",
        )
        with patch("asyncio.create_subprocess_shell", side_effect=OSError("no such file")):
            with pytest.raises(ExecutionError, match="Failed to run"):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_rollback_with_command(self):
        runner = PackageRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_INSTALL,
            description="Install vim",
            command="apt install -y vim",
            target="vim",
            rollback_command="echo 'removed'",
        )
        result = await runner.rollback(action)
        assert result.success is True
        assert result.rolled_back is True

    @pytest.mark.asyncio
    async def test_rollback_no_command(self):
        runner = PackageRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_UPGRADE,
            description="Upgrade",
            command="apt upgrade -y",
            target="system",
            rollback_command="",
        )
        result = await runner.rollback(action)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_rollback_command_fails(self):
        runner = PackageRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_INSTALL,
            description="Install",
            command="echo ok",
            target="pkg",
            rollback_command="false",
        )
        result = await runner.rollback(action)
        assert result.success is False
        assert result.rolled_back is False

    @pytest.mark.asyncio
    async def test_rollback_os_error(self):
        runner = PackageRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_INSTALL,
            description="Install",
            command="echo ok",
            target="pkg",
            rollback_command="bad_command",
        )
        with patch("asyncio.create_subprocess_shell", side_effect=OSError("fail")):
            with pytest.raises(ExecutionError, match="Rollback failed"):
                await runner.rollback(action)


class TestMemoryRunner:
    @pytest.mark.asyncio
    async def test_drop_caches_dry_run(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.DROP_CACHES,
            description="Drop caches",
            target="caches",
        )
        result = await runner.run(action, dry_run=True)
        assert result.success is True
        assert "DRY RUN" in result.output

    @pytest.mark.asyncio
    async def test_drop_caches_success(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.DROP_CACHES,
            description="Drop caches",
            command="echo ok",
            target="caches",
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"ok", b"")
        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await runner.run(action)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_drop_caches_failure(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.DROP_CACHES,
            description="Drop caches",
            target="caches",
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"permission denied")
        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            with pytest.raises(ExecutionError, match="Drop caches failed"):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_drop_caches_os_error(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.DROP_CACHES,
            description="Drop caches",
            target="caches",
        )
        with patch("asyncio.create_subprocess_shell", side_effect=OSError("fail")):
            with pytest.raises(ExecutionError, match="Failed to drop caches"):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_kill_by_memory_dry_run(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill hogs",
            target="chrome",
        )
        with patch.object(runner, "_find_memory_hogs", return_value=[(123, "chrome", 600.0)]):
            result = await runner.run(action, dry_run=True)
        assert result.success is True
        assert "DRY RUN" in result.output

    @pytest.mark.asyncio
    async def test_kill_by_memory_no_hogs(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill hogs",
            target="nonexistent",
        )
        with patch.object(runner, "_find_memory_hogs", return_value=[]):
            result = await runner.run(action)
        assert result.success is True
        assert "No memory-hogging" in result.output

    @pytest.mark.asyncio
    async def test_kill_by_memory_success(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill chrome",
            target="chrome",
        )
        mock_proc = MagicMock()
        with (
            patch.object(runner, "_find_memory_hogs", return_value=[(123, "chrome", 800.0)]),
            patch("agentic.executor.runners.memory_runner.psutil.Process", return_value=mock_proc),
        ):
            result = await runner.run(action)
        mock_proc.terminate.assert_called_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_kill_by_memory_access_denied(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_BY_MEMORY,
            description="Kill chrome",
            target="chrome",
        )
        mock_proc = MagicMock()
        mock_proc.terminate.side_effect = psutil.AccessDenied(pid=123)
        with (
            patch.object(runner, "_find_memory_hogs", return_value=[(123, "chrome", 800.0)]),
            patch("agentic.executor.runners.memory_runner.psutil.Process", return_value=mock_proc),
        ):
            with pytest.raises(ExecutionError, match="Failed to kill"):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_unsupported_action_type(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.APT_INSTALL,
            description="Wrong runner",
            target="pkg",
        )
        with pytest.raises(ExecutionError, match="cannot handle"):
            await runner.run(action)

    @pytest.mark.asyncio
    async def test_rollback_not_supported(self):
        runner = MemoryRunner()
        action = ActionCandidate(
            action_type=ActionType.DROP_CACHES,
            description="Drop",
            target="caches",
        )
        result = await runner.rollback(action)
        assert result.success is False
        assert result.rolled_back is False

    def test_find_memory_hogs_specific_target(self):
        mock_proc = MagicMock()
        mock_proc.info = {
            "pid": 100,
            "name": "chrome",
            "memory_info": MagicMock(rss=600 * 1024 * 1024),
        }
        mock_proc.pid = 100
        with patch("agentic.executor.runners.memory_runner.psutil.process_iter", return_value=[mock_proc]):
            hogs = MemoryRunner._find_memory_hogs("chrome")
        assert len(hogs) == 1

    def test_find_memory_hogs_generic(self):
        mock_proc = MagicMock()
        mock_proc.info = {
            "pid": 100,
            "name": "bigproc",
            "memory_info": MagicMock(rss=600 * 1024 * 1024),
        }
        mock_proc.pid = 100
        with patch("agentic.executor.runners.memory_runner.psutil.process_iter", return_value=[mock_proc]):
            hogs = MemoryRunner._find_memory_hogs("memory_hogs")
        assert len(hogs) == 1

    def test_find_memory_hogs_specific_target_no_match(self):
        mock_proc = MagicMock()
        mock_proc.info = {
            "pid": 100,
            "name": "firefox",
            "memory_info": MagicMock(rss=600 * 1024 * 1024),
        }
        mock_proc.pid = 100
        with patch("agentic.executor.runners.memory_runner.psutil.process_iter", return_value=[mock_proc]):
            hogs = MemoryRunner._find_memory_hogs("chrome")
        assert len(hogs) == 0

    def test_find_memory_hogs_below_threshold(self):
        mock_proc = MagicMock()
        mock_proc.info = {
            "pid": 100,
            "name": "smallproc",
            "memory_info": MagicMock(rss=100 * 1024 * 1024),
        }
        mock_proc.pid = 100
        with patch("agentic.executor.runners.memory_runner.psutil.process_iter", return_value=[mock_proc]):
            hogs = MemoryRunner._find_memory_hogs("memory_hogs")
        assert len(hogs) == 0

    def test_find_memory_hogs_no_memory_info(self):
        mock_proc = MagicMock()
        mock_proc.info = {"pid": 100, "name": "proc", "memory_info": None}
        mock_proc.pid = 100
        with patch("agentic.executor.runners.memory_runner.psutil.process_iter", return_value=[mock_proc]):
            hogs = MemoryRunner._find_memory_hogs("memory_hogs")
        assert len(hogs) == 0

    def test_find_memory_hogs_handles_exceptions(self):
        mock_proc = MagicMock()
        mock_proc.info.__getitem__ = MagicMock(side_effect=psutil.NoSuchProcess(1))
        # Make .get() also raise
        type(mock_proc).info = property(lambda self: (_ for _ in ()).throw(psutil.NoSuchProcess(1)))
        with patch("agentic.executor.runners.memory_runner.psutil.process_iter", return_value=[mock_proc]):
            hogs = MemoryRunner._find_memory_hogs("test")
        assert len(hogs) == 0


class TestSystemctlRunner:
    @pytest.mark.asyncio
    async def test_dry_run(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_START,
            description="Start nginx",
            target="nginx",
        )
        result = await runner.run(action, dry_run=True)
        assert result.success is True
        assert "DRY RUN" in result.output
        assert "start" in result.output

    @pytest.mark.asyncio
    async def test_start_success(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_START,
            description="Start nginx",
            target="nginx",
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"", b"")
        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await runner.run(action)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_stop_success(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_STOP,
            description="Stop nginx",
            target="nginx",
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"stopped", b"")
        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await runner.run(action)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_command_failure(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_RESTART,
            description="Restart nginx",
            target="nginx",
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"unit not found")
        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            with pytest.raises(ExecutionError, match="failed"):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_os_error(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_START,
            description="Start nginx",
            target="nginx",
        )
        with patch("asyncio.create_subprocess_shell", side_effect=OSError("fail")):
            with pytest.raises(ExecutionError, match="systemctl failed"):
                await runner.run(action)

    @pytest.mark.asyncio
    async def test_rollback_start_does_stop(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_START,
            description="Start nginx",
            target="nginx",
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"ok", b"")
        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await runner.rollback(action)
        assert result.rolled_back is True

    @pytest.mark.asyncio
    async def test_rollback_stop_does_start(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_STOP,
            description="Stop nginx",
            target="nginx",
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"ok", b"")
        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await runner.rollback(action)
        assert result.rolled_back is True

    @pytest.mark.asyncio
    async def test_rollback_failure(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_START,
            description="Start nginx",
            target="nginx",
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"failed")
        with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
            result = await runner.rollback(action)
        assert result.success is False
        assert result.rolled_back is False

    @pytest.mark.asyncio
    async def test_rollback_os_error(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.SYSTEMCTL_START,
            description="Start nginx",
            target="nginx",
        )
        with patch("asyncio.create_subprocess_shell", side_effect=OSError("fail")):
            with pytest.raises(ExecutionError, match="Rollback failed"):
                await runner.rollback(action)

    @pytest.mark.asyncio
    async def test_rollback_unsupported_action_type(self):
        runner = SystemctlRunner()
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,  # not in _REVERSE map
            description="Kill",
            target="test",
        )
        result = await runner.rollback(action)
        assert result.success is False
        assert "No rollback" in result.error


class TestActionExecutor:
    @pytest.mark.asyncio
    async def test_execute_dry_run(self):
        executor = ActionExecutor()
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend firefox",
            target="firefox",
        )
        with patch.object(ProcessRunner, "run", new_callable=AsyncMock) as mock_run:
            from agentic.models.action import ActionResult
            mock_run.return_value = ActionResult(action_id=action.id, success=True, output="ok")
            result = await executor.execute(action, dry_run=True)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_many(self):
        executor = ActionExecutor()
        actions = [
            ActionCandidate(
                action_type=ActionType.SUSPEND_PROCESS,
                description="Suspend",
                target="firefox",
            ),
            ActionCandidate(
                action_type=ActionType.SUSPEND_PROCESS,
                description="Suspend",
                target="chrome",
            ),
        ]
        with patch.object(ProcessRunner, "run", new_callable=AsyncMock) as mock_run:
            from agentic.models.action import ActionResult
            mock_run.return_value = ActionResult(action_id="x", success=True, output="ok")
            results = await executor.execute_many(actions, dry_run=True)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_execute_rollback(self):
        executor = ActionExecutor()
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Suspend",
            target="firefox",
        )
        with patch.object(ProcessRunner, "rollback", new_callable=AsyncMock) as mock_rb:
            from agentic.models.action import ActionResult
            mock_rb.return_value = ActionResult(action_id=action.id, success=True, rolled_back=True)
            result = await executor.rollback(action)
        assert result.rolled_back is True

    @pytest.mark.asyncio
    async def test_unregistered_action_type_raises(self):
        executor = ActionExecutor()
        action = ActionCandidate(
            action_type=ActionType.KILL_PROCESS,
            description="Test",
            target="test",
        )
        # Patch _RUNNER_MAP to remove the action type
        import agentic.executor.action_executor as mod
        original_map = mod._RUNNER_MAP.copy()
        mod._RUNNER_MAP.clear()
        try:
            with pytest.raises(ExecutionError, match="No runner registered"):
                await executor.execute(action)
        finally:
            mod._RUNNER_MAP.update(original_map)

    @pytest.mark.asyncio
    async def test_runner_caching(self):
        executor = ActionExecutor()
        action = ActionCandidate(
            action_type=ActionType.SUSPEND_PROCESS,
            description="Test",
            target="test",
        )
        runner1 = executor._get_runner(ActionType.SUSPEND_PROCESS)
        runner2 = executor._get_runner(ActionType.SUSPEND_PROCESS)
        assert runner1 is runner2
