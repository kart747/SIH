#!/usr/bin/env python3
"""Send synthetic IMU and GNSS samples to local ingest endpoint to trigger fail-safe."""
import requests
import time

URL = "http://127.0.0.1:8000/ingest"
DEVICE_ID = "test-device-1"

def send_step(vel_forward_m_s: float):
    payload = {
        "device_id": DEVICE_ID,
        "imu": {
            "acc": [0.0, 0.0, 0.0],
            "gyro": [0.0, 0.0, 0.0],
            "dt": 0.02
        },
        "gnss": {
            # send direct NED position and velocity (relative to origin)
            "pos_ned": [0.0, 0.0, 0.0],
            "vel_ned": [vel_forward_m_s, 0.0, 0.0]
        }
    }
    resp = requests.post(URL, json=payload, timeout=5.0)
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


def main():
    # Normal driving sequence: 5 m/s for 6 steps
    for i in range(6):
        send_step(5.0)
        time.sleep(0.1)

    # Inject sudden GNSS velocity jump to 20 m/s to trigger the 2 m/s fail-safe
    print("-- injecting GNSS velocity jump to 20 m/s --")
    send_step(20.0)
    time.sleep(0.1)

    # Return to normal
    for i in range(3):
        send_step(5.0)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
