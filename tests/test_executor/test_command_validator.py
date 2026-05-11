"""Tests for CommandValidator — full branch coverage."""

from __future__ import annotations

import pytest

from agentic.executor.command_validator import CommandValidator, ValidationResult, _DANGEROUS
from agentic.models.action import ActionCandidate, ActionType


def _action(command: str = "", target: str = "") -> ActionCandidate:
    return ActionCandidate(
        action_type=ActionType.KILL_PROCESS,
        description="test",
        command=command,
        target=target,
    )


class TestCommandValidatorCleanInputs:
    def test_empty_command_passes(self):
        v = CommandValidator()
        result = v.validate(_action())
        assert result.valid is True
        assert result.reason == ""

    def test_safe_kill_command_passes(self):
        v = CommandValidator()
        result = v.validate(_action(command="kill -15 1234", target="firefox"))
        assert result.valid is True

    def test_safe_systemctl_passes(self):
        v = CommandValidator()
        result = v.validate(_action(command="systemctl restart nginx", target="nginx"))
        assert result.valid is True

    def test_safe_apt_install_passes(self):
        v = CommandValidator()
        result = v.validate(_action(command="apt install -y vim", target="vim"))
        assert result.valid is True


class TestCommandValidatorDangerousCommands:
    def test_rm_rf_root_in_command(self):
        v = CommandValidator()
        result = v.validate(_action(command="rm -rf /"))
        assert result.valid is False
        assert "rm -rf /" in result.reason

    def test_rm_rf_root_variant(self):
        v = CommandValidator()
        result = v.validate(_action(command="rm -fr /tmp/../"))
        assert result.valid is False

    def test_raw_block_device_write_in_command(self):
        v = CommandValidator()
        result = v.validate(_action(command="cat /dev/urandom > /dev/sda"))
        assert result.valid is False
        assert "block device" in result.reason

    def test_dd_to_disk_in_command(self):
        v = CommandValidator()
        result = v.validate(_action(command="dd if=/dev/zero of=/dev/sda bs=4M"))
        assert result.valid is False
        assert "dd" in result.reason

    def test_mkfs_in_command(self):
        v = CommandValidator()
        result = v.validate(_action(command="mkfs.ext4 /dev/sdb1"))
        assert result.valid is False
        assert "format" in result.reason

    def test_shred_in_command(self):
        v = CommandValidator()
        result = v.validate(_action(command="shred -u /etc/passwd"))
        assert result.valid is False
        assert "shred" in result.reason

    def test_fork_bomb_in_command(self):
        v = CommandValidator()
        result = v.validate(_action(command=":(){ :|:& };:"))
        assert result.valid is False
        assert "fork bomb" in result.reason

    def test_chained_rm_with_pipe(self):
        # Use rm -r (no 'f' flag) so pattern 0 (rm -rf /) doesn't fire first
        v = CommandValidator()
        result = v.validate(_action(command="ls /tmp | rm -r /home/user"))
        assert result.valid is False
        assert "chained" in result.reason

    def test_chained_rm_with_semicolon(self):
        v = CommandValidator()
        result = v.validate(_action(command="ls; rm -rf /var/log"))
        assert result.valid is False


class TestCommandValidatorDangerousTargets:
    def test_dangerous_pattern_in_target_not_command(self):
        # command is clean, target has dangerous pattern
        v = CommandValidator()
        result = v.validate(_action(command="echo hello", target="shred"))
        assert result.valid is False

    def test_mkfs_in_target(self):
        v = CommandValidator()
        result = v.validate(_action(command="", target="mkfs.ext4"))
        assert result.valid is False

    def test_clean_target_passes(self):
        v = CommandValidator()
        result = v.validate(_action(command="kill -9 1234", target="chrome"))
        assert result.valid is True


class TestCommandValidatorValidateMany:
    def test_empty_list(self):
        v = CommandValidator()
        results = v.validate_many([])
        assert results == []

    def test_all_safe(self):
        v = CommandValidator()
        actions = [
            _action("kill -15 123", "firefox"),
            _action("systemctl stop nginx", "nginx"),
        ]
        results = v.validate_many(actions)
        assert len(results) == 2
        assert all(vr.valid for _, vr in results)

    def test_mixed_safe_and_unsafe(self):
        v = CommandValidator()
        safe = _action("kill -15 123", "firefox")
        unsafe = _action("rm -rf /", "")
        results = v.validate_many([safe, unsafe])
        assert results[0][1].valid is True
        assert results[1][1].valid is False

    def test_returns_correct_action_references(self):
        v = CommandValidator()
        action = _action("kill -15 999", "test")
        results = v.validate_many([action])
        returned_action, _ = results[0]
        assert returned_action is action


class TestCommandValidatorSemanticPatterns:
    """Semantic safety patterns — dangerous by effect, not by obvious syntax."""

    def test_find_delete_on_root(self):
        v = CommandValidator()
        result = v.validate(_action(command="find / -delete"))
        assert result.valid is False
        assert "find -delete" in result.reason

    def test_find_double_dash_delete_on_root(self):
        v = CommandValidator()
        result = v.validate(_action(command="find / --delete"))
        assert result.valid is False
        assert "find -delete" in result.reason

    def test_find_delete_on_root_with_args(self):
        v = CommandValidator()
        result = v.validate(_action(command="find / -type f -name '*.log' -delete"))
        assert result.valid is False

    def test_find_exec_rm_on_root(self):
        v = CommandValidator()
        result = v.validate(_action(command="find / -name '*.tmp' -exec rm {} \\;"))
        assert result.valid is False
        assert "find -exec rm" in result.reason

    def test_find_delete_on_subdirectory_passes(self):
        # find /home/user -delete is not covered — only / root is blocked
        v = CommandValidator()
        result = v.validate(_action(command="find /home/user -delete"))
        assert result.valid is True

    def test_chmod_recursive_world_writable_on_root(self):
        v = CommandValidator()
        result = v.validate(_action(command="chmod -R 777 /"))
        assert result.valid is False
        assert "chmod" in result.reason

    def test_chmod_recursive_world_writable_variant(self):
        v = CommandValidator()
        result = v.validate(_action(command="chmod -R 0777 /"))
        assert result.valid is False

    def test_chmod_recursive_safe_on_subdir_passes(self):
        v = CommandValidator()
        result = v.validate(_action(command="chmod -R 755 /var/www"))
        assert result.valid is True

    def test_chown_recursive_on_etc(self):
        v = CommandValidator()
        result = v.validate(_action(command="chown -R attacker /etc"))
        assert result.valid is False
        assert "chown" in result.reason

    def test_chown_recursive_on_boot(self):
        v = CommandValidator()
        result = v.validate(_action(command="chown -R nobody /boot"))
        assert result.valid is False

    def test_chown_recursive_on_usr(self):
        v = CommandValidator()
        result = v.validate(_action(command="chown -R user /usr"))
        assert result.valid is False

    def test_chown_recursive_on_home_passes(self):
        v = CommandValidator()
        result = v.validate(_action(command="chown -R user /home/user"))
        assert result.valid is True

    def test_rm_etc_passwd(self):
        v = CommandValidator()
        result = v.validate(_action(command="rm /etc/passwd"))
        assert result.valid is False
        assert "critical system file" in result.reason

    def test_rm_etc_shadow(self):
        v = CommandValidator()
        result = v.validate(_action(command="rm -f /etc/shadow"))
        assert result.valid is False

    def test_rm_etc_sudoers(self):
        v = CommandValidator()
        result = v.validate(_action(command="rm /etc/sudoers"))
        assert result.valid is False

    def test_rm_boot_file(self):
        v = CommandValidator()
        result = v.validate(_action(command="rm /boot/vmlinuz"))
        assert result.valid is False

    def test_rm_etc_passwd_in_target(self):
        v = CommandValidator()
        result = v.validate(_action(command="echo test", target="rm /etc/passwd"))
        assert result.valid is False


class TestValidationResultDataclass:
    def test_frozen(self):
        vr = ValidationResult(valid=True, reason="ok")
        with pytest.raises((AttributeError, TypeError)):
            vr.valid = False  # type: ignore[misc]

    def test_default_reason(self):
        vr = ValidationResult(valid=True)
        assert vr.reason == ""

    def test_dangerous_patterns_non_empty(self):
        assert len(_DANGEROUS) > 0
        for pattern, reason in _DANGEROUS:
            assert reason
            assert pattern.pattern
