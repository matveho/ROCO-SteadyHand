# Wrist camera live check — 2026-09-28

Result: blocked before image acquisition. No left-wrist image was captured.

Connected to dexmate@192.168.50.20 (hostname vega-1u). Ran the existing tools/vega_wrists_probe.py with /usr/bin/python3 and /home/dexmate/miniconda3/bin/python3; both fail importing wrist_cameras.

Only /dev/video0 and /dev/video1 exist, identified as vi-output, zedx 9-0020 and 9-0028. /run/wrist-cameras and /opt/wrist-cameras are absent. The inspected loaded-module list contains no ISX031 module. This does not prove physical absence: capture-board wiring/power and the supported Sony driver still need checking.

The vendor checkout has a ZED X One wrist example. /etc/dexmate/dexsensor/default.toml contains disabled left/right ZED_X_ONE_GS wrist configurations (camera_id 1/2). These are not evidence that this robot has ZED wrist cameras; do not enable them as a substitute for the documented Sony ISX031 path.

Diagnostic output was saved onboard at /tmp/steadyhand-wrist-diagnostic.txt and retrieved with tools/copy_from_vega.ps1. The committed excerpt retains device discovery and both import failures; vendor example source and unrelated default configuration are omitted. Robot clock reported 2026-09-11 despite this session being 2026-09-28; do not use its wall clock as proof of capture date.

No Robot() construction, arm/head motion, service restart, driver installation, or boot configuration change was performed.

Required next action: ask the robot supplier to identify the actual wrist capture board, verify its power/connection, and provide the supported ISX031 driver plus WristCameras API installation/start procedure. Once both streams are live, identify left versus right by physically covering the left lens and observing which labeled stream changes. Then save a fresh image and metadata, offload, and commit it. Never label wrist_a as left without this check.

## Update: camera software staged after maintainer confirmation

The earlier missing-assets conclusion was superseded by a privileged full filesystem search: assets existed under /home/dexmate/ROCO-SteadyHand/runs/onsite-20260927-first-access/dexmate-setup/. Latest upstream 4be4d16140b25f730673d664b49c58280b07ef46 is now staged at /home/dexmate/dexmate-setup-4be4d16/ using tools/copy_to_vega.ps1.

User reported maintainer confirmation for official ZED Link 1.4.3 package SHA256 c6963e0b999557df6ffc9d62f8bb8ac00af15618e21b5fdd18fa3ae239b582ac (different from historical manifest). Installed successfully over 1.4.1.

Camera-only setup installed /opt/wrist-cameras, system Python .pth, tmpfiles and enabled wrist-camera-init.service. Built and re-extracted/verified target-derived patched initrd using upstream builder; added DexmateSetupHeadWrists boot entry. Original DEFAULT Stereolabs retained. No reboot per explicit user instruction: console recovery unavailable. No camera modules hot-unloaded. Gripper/CAN libraries and head dexsensor config were not replaced by the broad setup installer.

Backup before vendor install: /var/backups/steadyhand-wrists-1789172301/system-before.tgz (boot, modules, service/config files, dpkg status/package metadata and vendor executables). The Stereolabs fallback entry now uses upgraded vendor files, not an untouched 1.4.1 installation; the archive preserves the pre-upgrade files.

Patched initrd SHA256: bca27d701dbcf739e54a59b18e7a882c20a25bb7c1ca501cee4af492800deb36.
Overlay SHA256: 31bb4f8734f0e53ad7f1c82cae773ef6c1cfaa00cbfc10166f37753365dcec94.

Verified system Python API import succeeds. Service enabled but inactive. Probe now correctly refuses missing current-boot ready.json. Still no wrist image and no physical left/right mapping.

Next: with display/keyboard recovery available, select DexmateSetupHeadWrists at boot; inspect wrist-camera-init.service and /run/wrist-cameras/ready.json, then run /usr/bin/python3 tools/vega_wrists_probe.py. Check head RGB again after boot. Identify left stream by occluding left lens before labeling captured images. Do not assume reboot alone selects the new entry: DEFAULT remains Stereolabs.

## Update: successful combined boot and first images

Combined boot succeeded after explicit user authorization with recovery available. Wrist initialization succeeded; both cameras produced real 1920x1536 RGB images with increasing frame IDs. Operator identified displayed first image (wrist_a) as RIGHT and second (wrist_b) as LEFT. Saved images and mapping are in images/. Head RGB capture also passed after restarting the publisher. No arm motion. Prior staging-only status is superseded. DEFAULT now selects DexmateSetupHeadWrists.
