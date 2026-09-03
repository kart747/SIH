package com.example.sensorstream

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Self-contained on-device dead-reckoning engine.
 *
 * Approach (pedestrian dead reckoning):
 *   1. Step detection — accelerometer magnitude peak detection with hysteresis
 *   2. Heading estimation — low-pass filtered magnetometer-based azimuth
 *   3. Position integration — each detected step advances (x, y) by
 *      stepLength * (cos(heading), sin(heading))
 *
 * All computation is in local NED-like frame (x = North, y = East) in metres.
 * No network or server dependency.
 */
class DeadReckoningEngine(
    /** Average step length in metres. 0.75 m is a reasonable adult default. */
    private val stepLength: Double = 0.75
) {

    // --- Public output state --------------------------------------------------

    /** Current position in metres: x = North, y = East, relative to start. */
    var posX: Double = 0.0
        private set
    var posY: Double = 0.0
        private set

    /** Current heading in degrees [0, 360) — 0 = North, 90 = East. */
    var headingDeg: Double = 0.0
        private set

    /** Estimated speed in m/s (simple moving average over recent steps). */
    var speedMps: Double = 0.0
        private set

    /** Total step count since start / last reset. */
    var stepCount: Int = 0
        private set

    // --- Step detection state -------------------------------------------------

    // We detect a step as a peak in |acc| that crosses above STEP_THRESHOLD_HIGH
    // and then falls below STEP_THRESHOLD_LOW (hysteresis prevents double-counts).
    private companion object {
        const val STEP_THRESHOLD_HIGH = 11.5   // m/s²  (gravity ≈ 9.81)
        const val STEP_THRESHOLD_LOW  = 9.0    // m/s²
        const val MIN_STEP_INTERVAL_NS = 250_000_000L  // 250 ms between steps minimum
        const val GRAVITY = 9.81

        // Low-pass filter coefficient for heading (closer to 1 = smoother)
        const val HEADING_ALPHA = 0.15

        // Speed estimation window
        const val SPEED_WINDOW = 5
    }

    private var aboveThreshold = false
    private var lastStepTimeNs: Long = 0L

    // --- Heading estimation state ---------------------------------------------

    private var headingRad: Double = 0.0
    private var headingInitialised = false

    // Gravity vector estimate (low-pass from accelerometer)
    private val gravity = DoubleArray(3)
    private var gravityInitialised = false

    // --- Speed estimation -----------------------------------------------------

    private val recentStepTimesNs = ArrayDeque<Long>(SPEED_WINDOW + 1)

    // =========================================================================
    //  Public API
    // =========================================================================

    /**
     * Feed a new accelerometer sample.  Call from onSensorChanged for TYPE_ACCELEROMETER.
     * @param values  [ax, ay, az] in m/s² (device frame, includes gravity)
     * @param timestampNs  SensorEvent.timestamp (nanoseconds, monotonic)
     */
    fun onAccelerometer(values: FloatArray, timestampNs: Long) {
        // Low-pass filter to estimate gravity direction
        val alpha = 0.8
        gravity[0] = alpha * gravity[0] + (1 - alpha) * values[0]
        gravity[1] = alpha * gravity[1] + (1 - alpha) * values[1]
        gravity[2] = alpha * gravity[2] + (1 - alpha) * values[2]
        gravityInitialised = true

        // Acceleration magnitude (includes gravity)
        val mag = sqrt(
            values[0].toDouble() * values[0] +
            values[1].toDouble() * values[1] +
            values[2].toDouble() * values[2]
        )

        detectStep(mag, timestampNs)
    }

    /**
     * Feed a new magnetometer sample.  Call from onSensorChanged for TYPE_MAGNETIC_FIELD.
     * Uses tilt-compensated compass heading derived from gravity + mag vectors.
     * @param values  [mx, my, mz] in µT (device frame)
     */
    fun onMagnetometer(values: FloatArray) {
        if (!gravityInitialised) return  // need gravity estimate first

        // Tilt-compensated azimuth calculation
        // Reference: Android SensorManager.getRotationMatrix / getOrientation logic
        val gx = gravity[0]; val gy = gravity[1]; val gz = gravity[2]
        val mx = values[0].toDouble(); val my = values[1].toDouble(); val mz = values[2].toDouble()

        // Cross product: East = M × G  (right-hand)
        val ex = my * gz - mz * gy
        val ey = mz * gx - mx * gz
        val ez = mx * gy - my * gx
        val eMag = sqrt(ex * ex + ey * ey + ez * ez)
        if (eMag < 1e-6) return  // degenerate

        val enx = ex / eMag; val eny = ey / eMag; val enz = ez / eMag

        // North = G × East
        val nx = gy * enz - gz * eny
        val _ny = gz * enx - gx * enz
        // val nz = gx * eny - gy * enx  // not needed for azimuth

        // Azimuth = atan2(dot(M_horizontal, East), dot(M_horizontal, North))
        // Simplified: azimuth from North/East projections directly
        val rawAzimuth = atan2(enx, nx)   // radians, 0 = North, positive = East

        // Low-pass filter the heading to smooth jitter
        if (!headingInitialised) {
            headingRad = rawAzimuth
            headingInitialised = true
        } else {
            // Circular low-pass: work in sin/cos domain to avoid wrap-around discontinuity
            val sinH = sin(headingRad) * (1 - HEADING_ALPHA) + sin(rawAzimuth) * HEADING_ALPHA
            val cosH = cos(headingRad) * (1 - HEADING_ALPHA) + cos(rawAzimuth) * HEADING_ALPHA
            headingRad = atan2(sinH, cosH)
        }

        // Convert to degrees [0, 360)
        var deg = Math.toDegrees(headingRad)
        if (deg < 0) deg += 360.0
        headingDeg = deg
    }

    /**
     * Feed a rotation vector sample.  If the device provides TYPE_ROTATION_VECTOR,
     * this gives a more accurate heading than raw magnetometer alone.
     * @param values  rotation vector quaternion components from SensorEvent
     */
    fun onRotationVector(values: FloatArray) {
        // Convert rotation vector to rotation matrix
        val rotationMatrix = FloatArray(9)
        android.hardware.SensorManager.getRotationMatrixFromVector(rotationMatrix, values)

        // Get orientation: [azimuth, pitch, roll]
        val orientation = FloatArray(3)
        android.hardware.SensorManager.getOrientation(rotationMatrix, orientation)

        val rawAzimuth = orientation[0].toDouble()  // radians

        if (!headingInitialised) {
            headingRad = rawAzimuth
            headingInitialised = true
        } else {
            val sinH = sin(headingRad) * (1 - HEADING_ALPHA) + sin(rawAzimuth) * HEADING_ALPHA
            val cosH = cos(headingRad) * (1 - HEADING_ALPHA) + cos(rawAzimuth) * HEADING_ALPHA
            headingRad = atan2(sinH, cosH)
        }

        var deg = Math.toDegrees(headingRad)
        if (deg < 0) deg += 360.0
        headingDeg = deg
    }

    /** Reset all state to initial (origin at current position). */
    fun reset() {
        posX = 0.0
        posY = 0.0
        headingDeg = 0.0
        headingRad = 0.0
        headingInitialised = false
        speedMps = 0.0
        stepCount = 0
        aboveThreshold = false
        lastStepTimeNs = 0L
        gravity[0] = 0.0; gravity[1] = 0.0; gravity[2] = 0.0
        gravityInitialised = false
        recentStepTimesNs.clear()
    }

    // =========================================================================
    //  Internals
    // =========================================================================

    private fun detectStep(accMagnitude: Double, timestampNs: Long) {
        if (!aboveThreshold) {
            // Waiting for acceleration to rise above high threshold
            if (accMagnitude > STEP_THRESHOLD_HIGH) {
                aboveThreshold = true
            }
        } else {
            // Waiting for acceleration to fall below low threshold → step complete
            if (accMagnitude < STEP_THRESHOLD_LOW) {
                aboveThreshold = false

                // Enforce minimum interval between steps
                if (lastStepTimeNs == 0L ||
                    (timestampNs - lastStepTimeNs) >= MIN_STEP_INTERVAL_NS
                ) {
                    onStepDetected(timestampNs)
                    lastStepTimeNs = timestampNs
                }
            }
        }
    }

    private fun onStepDetected(timestampNs: Long) {
        stepCount++

        // Advance position in the current heading direction
        // headingRad: 0 = North (+X), π/2 = East (+Y)
        posX += stepLength * cos(headingRad)
        posY += stepLength * sin(headingRad)

        // Update speed estimate from recent step cadence
        recentStepTimesNs.addLast(timestampNs)
        while (recentStepTimesNs.size > SPEED_WINDOW) {
            recentStepTimesNs.removeFirst()
        }
        speedMps = if (recentStepTimesNs.size >= 2) {
            val dtSec = (recentStepTimesNs.last() - recentStepTimesNs.first()).toDouble() / 1_000_000_000.0
            if (dtSec > 0) {
                (recentStepTimesNs.size - 1) * stepLength / dtSec
            } else 0.0
        } else 0.0
    }
}
