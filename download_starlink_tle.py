#!/usr/bin/env python3
# ===============================================
# 📄 download_starlink_tle.py
# Purpose: Download Starlink TLE from CelesTrak
# ===============================================

import os
import requests

# مسیر ذخیره فایل TLE
os.makedirs("data", exist_ok=True)
output_path = "data/starlink.txt"

# لینک رسمی TLE استارلینک در CelesTrak
tle_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=TLE"

print("[INFO] Downloading Starlink TLE from CelesTrak...")
response = requests.get(tle_url, timeout=15)

if response.status_code == 200 and len(response.text) > 0:
    with open(output_path, "w") as f:
        f.write(response.text)
    print(f"[✓] TLE file saved to: {output_path}")
    print(f"[✓] Number of lines: {len(response.text.splitlines())}")
else:
    print(f"[✗] Failed to download TLE. Status code: {response.status_code}")
    print(response.text[:200])
