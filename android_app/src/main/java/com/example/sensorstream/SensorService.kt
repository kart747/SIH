package com.example.sensorstream

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class SensorService : Service(), SensorEventListener {

    companion object {
        const val ACTION_STATUS_UPDATE = "com.example.sensorstream.STATUS_UPDATE"
        const val ACTION_DR_UPDATE = "com.example.sensorstream.DR_UPDATE"
        const val EXTRA_STATUS = "status"
        const val EXTRA_POS_X = "pos_x"
        const val EXTRA_POS_Y = "pos_y"
        const val EXTRA_HEADING = "heading"
        const val EXTRA_SPEED = "speed"
        const val EXTRA_STEPS = "steps"

        private const val TAG = "SensorService"
        private const val CHANNEL_ID = "sensor_stream_channel"
        private const val NOTIFICATION_ID = 42
        private const val DEVICE_ID = "android-device-1"
        private const val IMU_INTERVAL_MS = 20L

        // Server streaming is disabled by default for on-device demo.
        // Set to true and update SERVER_URL to your machine's IP to enable.
        private const val SERVER_STREAMING_ENABLED = false
        private const val SERVER_URL = "http://192.168.1.100:8000/ingest"
    }

    private lateinit var sensorManager: SensorManager
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var streamHandler: Handler
    private lateinit var streamThread: HandlerThread
    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(3, TimeUnit.SECONDS)
        .writeTimeout(3, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.SECONDS)
        .build()

    private val latestAcc = FloatArray(3)
    private val latestGyro = FloatArray(3)
    private val latestMag = FloatArray(3)
    private var latestLocation: android.location.Location? = null

    // On-device dead-reckoning engine
    private val drEngine = DeadReckoningEngine()

    // Throttle DR broadcasts to avoid flooding the UI (every ~100ms)
    private var lastDrBroadcastNs: Long = 0L
    private val drBroadcastIntervalNs = 100_000_000L  // 100 ms

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SensorManager::class.java)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        startForeground(NOTIFICATION_ID, buildNotification("Starting sensor stream..."))
        registerSensors()
        startLocationUpdates()
        if (SERVER_STREAMING_ENABLED) {
            startStreamingLoop()
        }
        publishStatus("Streaming")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(locationCallback)
        if (SERVER_STREAMING_ENABLED) {
            streamThread.quitSafely()
        }
        publishStatus("Stopped")
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAcc[0] = event.values[0]
                latestAcc[1] = event.values[1]
                latestAcc[2] = event.values[2]
                drEngine.onAccelerometer(event.values, event.timestamp)
            }
            Sensor.TYPE_GYROSCOPE -> {
                latestGyro[0] = event.values[0]
                latestGyro[1] = event.values[1]
                latestGyro[2] = event.values[2]
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                latestMag[0] = event.values[0]
                latestMag[1] = event.values[1]
                latestMag[2] = event.values[2]
                drEngine.onMagnetometer(event.values)
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                drEngine.onRotationVector(event.values)
            }
        }

        // Broadcast DR state at a throttled rate
        val now = event.timestamp
        if (now - lastDrBroadcastNs >= drBroadcastIntervalNs) {
            lastDrBroadcastNs = now
            publishDrUpdate()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun registerSensors() {
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        val rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME)
        } else {
            Log.w(TAG, "No accelerometer sensor available")
        }

        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME)
        } else {
            Log.w(TAG, "No gyroscope sensor available")
        }

        if (magnetometer != null) {
            sensorManager.registerListener(this, magnetometer, SensorManager.SENSOR_DELAY_GAME)
        } else {
            Log.w(TAG, "No magnetometer sensor available")
        }

        // Rotation vector gives better heading than raw mag when available
        if (rotationVector != null) {
            sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    private fun startLocationUpdates() {
        try {
            val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
                .setMinUpdateDistanceMeters(0f)
                .setGranularity(com.google.android.gms.location.Granularity.GRANULARITY_FINE)
                .setWaitForAccurateLocation(true)
                .build()

            fusedLocationClient.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
        } catch (e: SecurityException) {
            Log.e(TAG, "Location permission not granted", e)
            publishStatus("Location permission denied")
        }
    }

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(locationResult: LocationResult) {
            latestLocation = locationResult.lastLocation
        }
    }

    private fun startStreamingLoop() {
        streamThread = HandlerThread("sensor-stream-thread")
        streamThread.start()
        streamHandler = Handler(streamThread.looper)
        streamHandler.post(object : Runnable {
            override fun run() {
                sendDataPayload()
                streamHandler.postDelayed(this, IMU_INTERVAL_MS)
            }
        })
    }

    private fun sendDataPayload() {
        val payload = JSONObject().apply {
            put("device_id", DEVICE_ID)
            put("timestamp_ms", System.currentTimeMillis())
            put("imu", JSONObject().apply {
                put("acc", jsonArrayOf(latestAcc[0], latestAcc[1], latestAcc[2]))
                put("gyro", jsonArrayOf(latestGyro[0], latestGyro[1], latestGyro[2]))
                put("mag", jsonArrayOf(latestMag[0], latestMag[1], latestMag[2]))
                put("dt", 0.02)
            })

            if (latestLocation != null) {
                put("gnss", JSONObject().apply {
                    put("lat", latestLocation!!.latitude)
                    put("lon", latestLocation!!.longitude)
                    put("alt", latestLocation!!.altitude)
                    put("speed", latestLocation!!.speed.toDouble())
                    put("bearing", latestLocation!!.bearing.toDouble())
                })
            }
        }

        val requestBody = payload.toString().toRequestBody("application/json; charset=utf-8".toMediaType())
        val request = Request.Builder()
            .url(SERVER_URL)
            .post(requestBody)
            .build()

        okHttpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                publishStatus("Server: connection error")
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (response.isSuccessful) {
                        publishStatus("Server: connected")
                    } else {
                        publishStatus("Server error: ${response.code}")
                    }
                }
            }
        })
    }

    private fun jsonArrayOf(x: Float, y: Float, z: Float): JSONArray {
        return JSONArray().apply {
            put(x.toDouble())
            put(y.toDouble())
            put(z.toDouble())
        }
    }

    private fun buildNotification(contentText: String): Notification {
        createNotificationChannel()
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Dead Reckoning Sensor Stream")
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Sensor stream",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun publishStatus(message: String) {
        val intent = Intent(ACTION_STATUS_UPDATE)
        intent.setPackage(packageName)
        intent.putExtra(EXTRA_STATUS, message)
        sendBroadcast(intent)
    }

    private fun publishDrUpdate() {
        val intent = Intent(ACTION_DR_UPDATE).apply {
            setPackage(packageName)
            putExtra(EXTRA_POS_X, drEngine.posX)
            putExtra(EXTRA_POS_Y, drEngine.posY)
            putExtra(EXTRA_HEADING, drEngine.headingDeg)
            putExtra(EXTRA_SPEED, drEngine.speedMps)
            putExtra(EXTRA_STEPS, drEngine.stepCount)
        }
        sendBroadcast(intent)
    }
}
