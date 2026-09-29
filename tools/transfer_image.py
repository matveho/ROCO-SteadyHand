from pathlib import Path
from datetime import datetime
import subprocess

DEX = "dexmate@192.168.50.20"
LOCAL_DIR = Path.home() / "roco_wrist_images"

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

while True:
    try:
        remote_path = input("Dex path: ").strip().strip('"').strip("'")

        if not remote_path:
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = LOCAL_DIR / f"battery_size1_{timestamp}.png"

        source = f"{DEX}:{remote_path}"

        print(f"Downloading -> {destination}")

        result = subprocess.run([
            "scp",
            source,
            str(destination),
        ])

        if result.returncode == 0:
            print(f"OK: {destination}")
        else:
            print("SCP FAILED")

    except KeyboardInterrupt:
        print("\nExiting.")
        break