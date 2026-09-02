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
        const val EXTRA_STATUS = "status"

        private const val CHANNEL_ID = "sensor_stream_channel"
        private const val NOTIFICATION_ID = 42
        private const val DEVICE_ID = "android-device-1"
        private const val SERVER_URL = "http://<YOUR_LAPTOP_IP>:8000/ingest"
        private const val IMU_INTERVAL_MS = 20L
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

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(SensorManager::class.java)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        startForeground(NOTIFICATION_ID, buildNotification("Starting sensor stream..."))
        registerSensors()
        startLocationUpdates()
        startStreamingLoop()
        publishStatus("Streaming")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        fusedLocationClient.removeLocationUpdates(locationCallback)
        streamThread.quitSafely()
        publishStatus("Stopped")
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                latestAcc[0] = event.values[0]
                latestAcc[1] = event.values[1]
                latestAcc[2] = event.values[2]
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
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun registerSensors() {
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST)
        }

        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST)
        }

        if (magnetometer != null) {
            sensorManager.registerListener(this, magnetometer, SensorManager.SENSOR_DELAY_FASTEST)
        }
    }

    private fun startLocationUpdates() {
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
            .setMinUpdateDistanceMeters(0f)
            .setGranularity(com.google.android.gms.location.Granularity.GRANULARITY_FINE)
            .setWaitForAccurateLocation(true)
            .build()

        fusedLocationClient.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
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
                publishStatus("Connection error")
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (response.isSuccessful) {
                        publishStatus("Connected")
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
        intent.putExtra(EXTRA_STATUS, message)
        sendBroadcast(intent)
    }
}
