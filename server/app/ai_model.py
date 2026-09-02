import numpy as np
import hashlib


class AIMock:
    """Deterministic placeholder AI model that predicts velocity from IMU.

    This mock is deterministic: given device_id and recent inputs it returns a
    small velocity prediction. Replace with real model loading/inference.
    """

    def predict_velocity(self, device_id: str, acc: np.ndarray, gyro: np.ndarray, dt: float) -> list[float]:
        # Create a deterministic hash-based seed so outputs are reproducible
        h = hashlib.sha256(device_id.encode("utf-8") + acc.tobytes() + gyro.tobytes())
        seed = int.from_bytes(h.digest()[:4], "little")
        rng = np.random.RandomState(seed)

        # Simple estimation: integrate specific force magnitude over dt scaled
        speed_est = np.linalg.norm(acc) * dt * 0.5
        # generate direction from small random vector for demo
        direction = rng.randn(3)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        vel = direction * speed_est
        return vel.tolist()
