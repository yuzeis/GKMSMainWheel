# Copyright (C) 2026 花吹雪また一年P
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``python -m GakumasCLI <command> [args]`` (or ``GakumasCLI/gkms.py <command> [args]``): run one of the package's tools.
"""
from __future__ import annotations

import runpy
import sys

COMMANDS = {
    "master-bootstrap": "GakumasCLI.tools.master_bootstrap",
    "gen-tables": "GakumasCLI.native.gen_tables",
    "native-coverage": "GakumasCLI.native.native_coverage",
    "search-ab": "GakumasCLI.native.search_ab",
}


def module_of(command):
    if command in COMMANDS:
        return COMMANDS[command]
    name = command if command.startswith("GakumasCLI.") else "GakumasCLI." + command
    return name.removesuffix(".py").replace("/", ".")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        print("\ncommands:")
        for name, module in COMMANDS.items():
            print(f"  {name:18s} {module}")
        return 0
    module = module_of(argv[0])
    from GakumasCLI import paths
    sys.argv = [module, *argv[1:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
