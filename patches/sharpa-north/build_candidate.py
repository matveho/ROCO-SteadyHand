#!/usr/bin/env python3
"""Compile a copied North SDK's controller into a separate output directory.

Reads the copied CMake build settings; does not invoke CMake, install a plugin,
launch a controller, or modify SDK sources. Run on the SDK's build machine.
"""
import argparse
from pathlib import Path
import shlex
import subprocess


def relocate(arguments, original, copied):
    """Remap absolute build inputs to the SDK copy, preserving shell-free argv."""
    return [arg.replace(str(original), str(copied)).replace(r"\$ORIGIN", "$ORIGIN")
            for arg in arguments]


def build(sdk, output, original):
    sdk, output, original = sdk.resolve(), output.resolve(), original.resolve()
    if sdk == original or original in sdk.parents or sdk in original.parents:
        raise ValueError("Build SDK must be a separate copy of the original")
    if any(output == p or p in output.parents or output in p.parents
           for p in (sdk, original)):
        raise ValueError("Build output must be separate from both SDK trees")
    output.mkdir(exist_ok=False)
    directory = sdk / "build/src/plugins"
    settings = directory / "CMakeFiles/master_controller_plugin.dir"
    flags = {}
    for line in (settings / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = relocate(shlex.split(value), original, sdk)
    link = relocate(shlex.split((settings / "link.txt").read_text()), original, sdk)
    objects = {}
    for name in ("north_controller.cpp", "master_controller_plugin.cpp"):
        obj = output / (name + ".o")
        command = ["nice", "-n", "15", link[0]]
        for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
            command.extend(flags[key])
        command += ["-c", str(sdk / "src/plugins" / name), "-o", str(obj)]
        with (output / (name + ".build.log")).open("w") as log:
            subprocess.run(command, cwd=directory, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        objects[name + ".o"] = str(obj)
        print("Compiled:", name, flush=True)
    link = [objects.get(Path(arg).name, arg) for arg in link]
    binary = output / "master_controller_plugin.so"
    link[link.index("-o") + 1] = str(binary)
    with (output / "link.log").open("w") as log:
        subprocess.run(["nice", "-n", "15"] + link, cwd=directory,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True, help="Copied SDK")
    parser.add_argument("--original-sdk", type=Path, required=True,
                        help="Original path embedded in copied CMake metadata")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("Built candidate (not activated):", build(args.sdk, args.output, args.original_sdk))


if __name__ == "__main__":
    main()
