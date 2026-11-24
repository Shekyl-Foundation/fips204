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

Anything else fails: a forward branch or an overflow-panic branch. The caller
states how many back-edges and gamma2 tests this binary has
(`--expected aarch64|x86_64=BACK_EDGES,GAMMA2_TESTS`). The count is a property
of that binary and of one rustc: another binary inlines the same helpers in
more places, and a compiler change moves the number. This script has no count
of its own. A pair for an architecture other than the one being read is
accepted and not examined. The check fails if no helper is found (rule: a
gate asserts its subject exists).

Usage: check_branch_counts.py BINARY aarch64|x86_64 --expected ARCH=BACK,GAMMA2
"""

import argparse
import collections
import os
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
# A helper's closure is its own frame (`infinity_norm::{{closure}}`, `{closure#0}`,
# or `closure$1` from a PDB) and counts as the helper.
HELPER_NAME = re.compile(
    r"(?:^|::)(" + "|".join(HELPERS) + r")(?:<[^:]*>)?(?:$|::h[0-9a-f]{16}|::\{+closure|::closure)"
)
HELPER_FILE = re.compile(r"fips204[^ ]*/src/(?:helpers|high_low)\.rs:")
TOOLS = {
    "aarch64": ("aarch64-linux-gnu-objdump", "aarch64-linux-gnu-addr2line"),
    "x86_64": ("objdump", "addr2line"),
}
CONDITIONAL = {
    "aarch64": re.compile(r"^(b\.[a-z]+|cbz|cbnz|tbz|tbnz)$"),
    "x86_64": re.compile(r"^j(?!mp)[a-z]+$"),
}
# `decompose` branches on `gamma2 & (1 << 17)`, the public parameter that
# selects ML-DSA-44 against ML-DSA-65 and ML-DSA-87. On x86_64, rustc lowers
# that to `test $0x20000, %reg`.
GAMMA2_SET_BIT = 1 << 17
GAMMA2_TEST = re.compile(rf"^\$0x{GAMMA2_SET_BIT:x},%[a-z0-9]+$")


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


def expected_pair(text):
    """`aarch64=16,0` as `("aarch64", (16, 0))`."""
    match = re.fullmatch(r"(aarch64|x86_64)=(\d+),(\d+)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"{text!r} is not ARCH=BACK_EDGES,GAMMA2_TESTS")
    return match.group(1), (int(match.group(2)), int(match.group(3)))


def self_test():
    """The pair parser accepts one shape and refuses the others."""
    arch, pair = expected_pair("x86_64=16,4")
    if (arch, pair) != ("x86_64", (16, 4)):
        print("FAIL: self-test: a well-formed pair did not parse")
        return 1
    for bad in ("x86_64", "x86_64=16", "arm=1,0", "x86_64=16,4,1", "x86_64=-1,0"):
        try:
            expected_pair(bad)
        except argparse.ArgumentTypeError:
            continue
        print(f"FAIL: self-test: accepted {bad}")
        return 1
    print("PASS self-test")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv == ["--self-test"]:
        return self_test()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("binary")
    parser.add_argument("arch", choices=sorted(TOOLS))
    parser.add_argument("--expected", type=expected_pair, action="append", required=True)
    args = parser.parse_args(argv)
    expected = {}
    for arch, pair in args.expected:
        if arch in expected:
            parser.error(f"{arch} is stated twice")
        expected[arch] = pair
    if args.arch not in expected:
        parser.error(f"no --expected pair for {args.arch}")
    binary, arch = args.binary, args.arch
    if not os.path.isfile(binary):
        print(f"FAIL: no binary at {binary}")
        return 1
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
    if (backs, gammas) != expected[arch]:
        print(f"FAIL: expected (back-edges, gamma2 tests) {expected[arch]}, found {(backs, gammas)}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
