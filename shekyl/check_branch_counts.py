#!/usr/bin/env python3
# Copyright (c) 2026, The Shekyl Foundation
#
# All rights reserved.
# BSD-3-Clause
"""Exact count of conditional branches in fips204's coefficient helpers.

Reads a binary built with `overflow-checks = true` and line tables
(`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`) and attributes every
conditional branch to the source frames inlined at it, so a helper inlined
into `sign_internal` is counted like a standalone one.

A branch counts when any frame at it is one of HELPERS in fips204's
`helpers.rs` or `high_low.rs`. It must be:
- a loop back-edge; or
- on x86_64, the test of the public parameter `gamma2`
  (`test $0x20000,<reg>` immediately before the jump).

Anything else fails: a forward branch or an overflow-panic branch. The number
of back-edges and gamma2 tests must equal EXPECTED exactly, so a compiler that
changes the shape of these functions fails until a person reads the result.
The counts are for rustc 1.94.0 and shekyl-core's example
`ca3_pq_sign_timing`. The check fails if no helper is found (rule: a gate
asserts its subject exists).

Usage: check_branch_counts.py BINARY aarch64|x86_64
"""

import collections
import re
import subprocess
import sys

HELPERS = (
    "partial_reduce32",
    "full_reduce32",
    "center_mod",
    "infinity_norm",
    "decompose",
    "high_bits",
    "low_bits",
    "make_hint",
)
HELPER_NAME = re.compile(r"(?:^|::)(" + "|".join(HELPERS) + r")(?:<|$|::h[0-9a-f]{16})")
HELPER_FILE = re.compile(r"fips204[^ ]*/src/(?:helpers|high_low)\.rs:")
TOOLS = {
    "aarch64": ("aarch64-linux-gnu-objdump", "aarch64-linux-gnu-addr2line"),
    "x86_64": ("objdump", "addr2line"),
}
CONDITIONAL = {
    "aarch64": re.compile(r"^(b\.[a-z]+|cbz|cbnz|tbz|tbnz)$"),
    "x86_64": re.compile(r"^j(?!mp)[a-z]+$"),
}
GAMMA2_TEST = re.compile(r"^\$0x20000,%[a-z0-9]+$")
# (loop back-edges, gamma2 tests), for the build named in the docstring.
EXPECTED = {"aarch64": (27, 0), "x86_64": (27, 4)}


def instructions(binary, arch):
    """Every `.text` function as (start, [(address, mnemonic, operands)])."""
    text = subprocess.run(
        [TOOLS[arch][0], "-d", "--no-show-raw-insn", "-j", ".text", binary],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    functions, current = [], None
    for line in text.splitlines():
        header = re.match(r"^([0-9a-f]+) <.+>:$", line)
        if header:
            current = (int(header.group(1), 16), [])
            functions.append(current)
            continue
        row = re.match(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if row and current is not None:
            current[1].append((int(row.group(1), 16), row.group(2), row.group(3).strip()))
    return functions


def frames(binary, arch, addresses):
    """{address: [(function, file:line)]}, innermost first."""
    result = {}
    for start in range(0, len(addresses), 2000):
        chunk = addresses[start : start + 2000]
        output = subprocess.run(
            [TOOLS[arch][1], "-a", "-f", "-i", "-C", "-e", binary] + [hex(a) for a in chunk],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        current, index = None, 0
        while index < len(output):
            if re.match(r"^0x[0-9a-f]+$", output[index]):
                current = int(output[index], 16)
                result[current] = []
                index += 1
                continue
            result[current].append((output[index], output[index + 1]))
            index += 2
    return result


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in TOOLS:
        print(__doc__)
        return 2
    binary, arch = sys.argv[1:]
    functions = instructions(binary, arch)
    branches = []
    for start, body in functions:
        end = body[-1][0] if body else start
        for position, (address, mnemonic, operands) in enumerate(body):
            if not CONDITIONAL[arch].match(mnemonic):
                continue
            target = re.search(r"\b([0-9a-f]+)(?: <|$)", operands)
            target = int(target.group(1), 16) if target else None
            previous = body[position - 1] if position else (0, "", "")
            gamma2 = previous[1] == "test" and bool(GAMMA2_TEST.match(previous[2]))
            back = target is not None and start <= target <= address <= end
            branches.append((address, back, gamma2))
    where = frames(binary, arch, [b[0] for b in branches])
    if not any(where.values()) or all(f[0][1].startswith("??") for f in where.values() if f):
        print("FAIL: the binary has no line tables")
        return 1
    backs, gammas, bad = 0, 0, collections.Counter()
    for address, back, gamma2 in branches:
        named = [
            f for f in where.get(address, []) if HELPER_NAME.search(f[0]) and HELPER_FILE.search(f[1])
        ]
        if not named:
            continue
        if back:
            backs += 1
        elif gamma2:
            gammas += 1
        else:
            function = HELPER_NAME.search(named[0][0]).group(1)
            bad[(function, named[0][1].rsplit("/", 1)[-1].split(" ")[0])] += 1
    print(f"{arch}: loop back-edges={backs} gamma2 tests={gammas} other={sum(bad.values())}")
    for (function, line), count in sorted(bad.items()):
        print(f"  not allowed: {function} {line} x{count}")
    if backs + gammas + sum(bad.values()) == 0:
        print("FAIL: no helper found in this binary")
        return 1
    if bad:
        print("FAIL: a conditional branch on a coefficient")
        return 1
    if (backs, gammas) != EXPECTED[arch]:
        print(f"FAIL: expected (back-edges, gamma2 tests) {EXPECTED[arch]}, found {(backs, gammas)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
