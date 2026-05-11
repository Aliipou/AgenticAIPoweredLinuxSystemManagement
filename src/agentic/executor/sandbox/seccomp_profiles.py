"""Per-ActionScope seccomp profiles — minimal syscall whitelists.

Each profile is derived from the ActionScope, not from the action instance.
The defaultAction is SCMP_ACT_ERRNO: any syscall not on the whitelist returns
EPERM to the process. This limits blast radius from command injection — an
injected command cannot make syscalls outside the scope's whitelist even if
it runs in the container.

These profiles target x86_64 Linux. They are not verified against every kernel
version or libc; treat them as a starting point, not a provably complete list.
"""

from __future__ import annotations

import json

from agentic.models.action import ActionScope

# Syscalls required by any dynamically-linked executable: loader, libc init,
# signal handling, memory allocation, basic I/O, process exit.
_BASELINE: list[str] = [
    "read", "write", "close",
    "exit", "exit_group",
    "brk", "mmap", "mmap2", "munmap", "mprotect", "mremap",
    "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "sigreturn",
    "rt_sigpending", "rt_sigsuspend",
    "fstat", "fstat64", "stat", "stat64", "lstat", "lstat64",
    "newfstatat", "statx",
    "uname",
    "getuid", "getuid32", "getgid", "getgid32",
    "geteuid", "geteuid32", "getegid", "getegid32",
    "getgroups", "getgroups32",
    "getpid", "getppid", "getpgid", "getsid",
    "arch_prctl", "set_tid_address", "set_robust_list", "get_robust_list",
    "futex", "futex_time64",
    "nanosleep", "clock_nanosleep", "clock_nanosleep_time64",
    "clock_gettime", "clock_gettime64", "clock_getres",
    "gettimeofday", "time",
    "pread64", "pwrite64", "readv", "writev", "preadv", "pwritev",
    "ioctl", "fcntl", "fcntl64",
    "lseek", "llseek", "_llseek",
    "dup", "dup2", "dup3",
    "pipe", "pipe2",
    "poll", "ppoll", "ppoll_time64",
    "select", "pselect6", "pselect6_time64",
    "epoll_create", "epoll_create1", "epoll_ctl", "epoll_wait", "epoll_pwait",
    "epoll_pwait2",
    "open", "openat", "openat2",
    "getcwd", "chdir", "fchdir",
    "getdents", "getdents64",
    "readlink", "readlinkat",
    "access", "faccessat", "faccessat2",
]

# Extra syscalls needed per scope, beyond the baseline.
_SCOPE_EXTRA: dict[ActionScope, list[str]] = {
    ActionScope.PROCESS: [
        # Signal delivery to host processes (requires --pid=host at runtime)
        "kill", "tgkill", "tkill",
        # Priority adjustment (renice)
        "setpriority", "getpriority",
        # Wait for child processes
        "wait4", "waitpid", "waitid",
    ],
    ActionScope.PACKAGE: [
        # apt needs to exec itself, dpkg, dpkg-deb, etc.
        "execve", "execveat",
        "fork", "clone", "clone3", "vfork",
        # Filesystem mutations (package install writes files)
        "mkdir", "mkdirat", "rmdir",
        "rename", "renameat", "renameat2",
        "unlink", "unlinkat",
        "link", "linkat", "symlink", "symlinkat",
        "chmod", "fchmod", "fchmodat",
        "chown", "fchown", "lchown", "fchownat",
        "truncate", "ftruncate",
        "mknod", "mknodat",
        "statfs", "fstatfs", "statfs64",
        # Network: apt fetches packages over HTTP/HTTPS
        "socket", "connect", "bind", "listen",
        "accept", "accept4",
        "send", "sendto", "sendmsg", "sendmmsg",
        "recv", "recvfrom", "recvmsg", "recvmmsg",
        "getsockopt", "setsockopt", "getsockname", "getpeername",
        "shutdown",
        # Process wait (apt runs multiple subprocesses)
        "wait4", "waitpid", "waitid",
    ],
    ActionScope.MEMORY: [
        # Write to /proc/sys/vm/drop_caches
        "statfs", "fstatfs", "statfs64",
        "sync", "syncfs", "fsync", "fdatasync",
    ],
    ActionScope.SERVICE: [
        # systemctl execs itself
        "execve", "execveat",
        "fork", "clone", "clone3", "vfork",
        # D-Bus socket communication
        "socket", "connect", "bind",
        "send", "sendto", "sendmsg",
        "recv", "recvfrom", "recvmsg",
        "getsockopt", "setsockopt", "getsockname", "getpeername",
        "shutdown",
        # Process wait
        "wait4", "waitpid", "waitid",
        # systemctl reads unit files
        "statfs", "fstatfs", "statfs64",
        "mkdir", "mkdirat",
        "rename", "renameat",
    ],
}


def build_seccomp_profile(scope: ActionScope) -> dict:
    """Return a Docker-compatible seccomp profile dict for the given scope.

    The profile whitelists exactly the syscalls needed for baseline execution
    plus the scope-specific set. All other syscalls return EPERM.
    """
    combined = list(dict.fromkeys(_BASELINE + _SCOPE_EXTRA.get(scope, [])))
    return {
        "defaultAction": "SCMP_ACT_ERRNO",
        "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86", "SCMP_ARCH_X32"],
        "syscalls": [{"names": combined, "action": "SCMP_ACT_ALLOW"}],
    }


def seccomp_json(scope: ActionScope) -> str:
    """Serialise a seccomp profile to JSON for use as a Docker --security-opt value."""
    return json.dumps(build_seccomp_profile(scope), separators=(",", ":"))
