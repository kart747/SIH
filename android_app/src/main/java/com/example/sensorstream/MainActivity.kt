package com.example.sensorstream

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import okhttp3.*
import org.json.JSONObject
import java.io.IOException

class MainActivity : AppCompatActivity(), SensorEventListener, LocationListener {
    private lateinit var sensorManager: SensorManager
    private var accelValues = floatArrayOf(0f, 0f, 0f)
    private var gyroValues = floatArrayOf(0f, 0f, 0f)

    private lateinit var locationManager: LocationManager
    private var lastLocation: Location? = null

    private val client = OkHttpClient()
    private val serverUrl = "http://10.0.2.2:8000/ingest" // emulator localhost mapping
    private val deviceId = "android-device-1"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager

        // request runtime permissions
        val requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { perms ->
            // ignore result for minimal demo
        }

        requestPermissionLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION))

        // register sensors
        sensorManager.registerListener(
            this,
            sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
            SensorManager.SENSOR_DELAY_GAME
        )
        sensorManager.registerListener(
            this,
            sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
            SensorManager.SENSOR_DELAY_GAME
        )

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0f, this)
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accelValues = event.values.clone()
                sendPayload()
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyroValues = event.values.clone()
                sendPayload()
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onLocationChanged(location: Location) {
        lastLocation = location
        sendPayload()
    }

    private fun sendPayload() {
        val imu = JSONObject()
        val acc = listOf(accelValues[0].toDouble(), accelValues[1].toDouble(), accelValues[2].toDouble())
        val gyro = listOf(gyroValues[0].toDouble(), gyroValues[1].toDouble(), gyroValues[2].toDouble())
        imu.put("acc", acc)
        imu.put("gyro", gyro)
        imu.put("dt", 0.02)

        val payload = JSONObject()
        payload.put("device_id", deviceId)
        payload.put("imu", imu)

        if (lastLocation != null) {
            val gnss = JSONObject()
            // This demo sends lat/lon/alt; server expects pos_ned for accurate fusion.
            gnss.put("lat", lastLocation!!.latitude)
            gnss.put("lon", lastLocation!!.longitude)
            gnss.put("alt", lastLocation!!.altitude)
            gnss.put("speed", lastLocation!!.speed.toDouble())
            payload.put("gnss", gnss)
        }

        val body = RequestBody.create(MediaType.get("application/json; charset=utf-8"), payload.toString())
        val request = Request.Builder().url(serverUrl).post(body).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                // ignore for demo
            }

            override fun onResponse(call: Call, response: Response) {
                response.close()
            }
        })
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
    }
}
