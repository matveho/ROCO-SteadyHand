# North arm-index fix and separate builds

This folder is intended for the public SteadyHand repository. It contains only
the patch, a baseline checksum manifest, offline tests, and build helpers. Keep
vendor SDK trees, binaries, recordings, credentials and machine-specific settings
outside Git. A public branch is public too.

## What the patch fixes

`NorthController::extractActions()` reverses seven arm positions into arrays
whose valid indexes are 0 through 6. The original `7 - i` starts at index 7 and
leaves index 0 unwritten. The patch uses `6 - i` and rejects any present arm that
does not have exactly seven finite positions before any bundle is published.
Omitted arm fields retain their existing behavior.

The actual production methods reproduced a stack-buffer-overflow before the fix
and passed 271 offline cases under AddressSanitizer and UndefinedBehaviorSanitizer
after it. These tests use real SDK protobuf classes with offline transport and
stability-checker doubles. They never instantiate a hardware controller.

This is an indexing correction. It does not change the robot's seven degrees of
freedom, gains, payload, homing tolerance, timeout, or stop behavior. It does not
establish physical tracking accuracy or resolve angular tracking errors.

## Copy, patch, test and build

Run on the machine with the organizer's complete SDK and C++ build dependencies.
The baseline is the SDK labeled 1.6.3, identified by the exact hashes in
`baseline.json`. A different source or original plugin is rejected, even if its
version label is the same. Review/retest a new baseline instead of bypassing this
check. Required tools: Python 3.10+, GNU patch, C++ compiler with sanitizers,
protobuf development libraries, readelf, and the original SDK's build dependencies.

From the repository root, substitute your actual SDK and a new workspace path:

```bash
python3 -B patches/sharpa-north/prepare_sdk.py \
  --sdk /path/to/original-sdk \
  --workspace /path/outside-git/north-sdk-workspaces/arm-index-v1 \
  --build
```

Without `--build`, the helper only copies and patches sources. With it, the helper
runs the 271 offline cases and compiles a candidate plugin. It:

1. Checks the original header, source and plugin against the baseline hashes.
2. Rejects existing workspaces, destinations inside Git or the original SDK, and
   SDK symlinks that escape the source tree. It does not resolve external links by
   guessing; review those dependencies separately.
3. Copies the SDK using independent files, excluding `.git`, `.backups`, logs,
   recordings, loose log files, motor memory dumps and Python caches. Internal
   absolute symlinks are relocated into the copy. No hardlinks connect the copy
   to the original. A fresh empty `logs/` directory is created for North's logger.
4. Applies the patch without fuzzy matching and verifies the resulting source
   hashes. Compiler/linker paths referring to the original SDK are remapped to
   the copy in memory. Original build files remain unchanged.
5. Builds into `workspace/build/` and writes a local `receipt.json`. The candidate
   plugin stays outside the copied SDK's plugin directory; nothing is activated.
6. Rechecks the three original baseline files before reporting success.

A failed workspace is left for inspection and cannot be reused accidentally.
The manifest pins the changed source/header and original controller plugin, not
every vendor dependency; a matching baseline is not proof of runtime compatibility.

## Activation and rollback

Build success is an offline result. The helper deliberately has no installation,
startup, shutdown, service-query or movement operation. Native launch scripts can
contain absolute SDK paths, and existing executables can contain absolute library
search paths. Those must be inspected on the target machine before using a copied
runtime. Do not point the vendor launcher at the original tree and assume it loads
the candidate.

For a reviewed deployment, stop the existing controller through its normal
shutdown, check that no controller remains, put the candidate plugin only into
the separate SDK copy, and launch that copy with verified configuration/library
paths. Verify the running process actually maps the candidate plugin and starts
in its expected initial state before sending motion commands. Run one controller
at a time. Keep the existing original launch path for rollback; rollback switches
back to that path rather than rewriting the original files. Runtime activation
and physical motion are not automated or claimed verified by this package.

## Tests without the vendor SDK

```bash
python3 -B -m unittest discover -s tests -p 'test_north_patch_workflow.py' -v
```

These fixture tests verify the workflow's write boundaries, baseline rejection,
symlink handling and source preservation. The C++ production-method regression
requires the SDK and runs during `--build`. Do not describe fixture tests as a
fresh hardware or production C++ validation.
