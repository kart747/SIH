package com.example.sensorstream

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var connectionStatusText: TextView
    private lateinit var positionXText: TextView
    private lateinit var positionYText: TextView
    private lateinit var headingText: TextView
    private lateinit var speedText: TextView
    private lateinit var stepCountText: TextView
    private lateinit var permissionLauncher: androidx.activity.result.ActivityResultLauncher<Array<String>>

    // BroadcastReceiver for status updates from SensorService
    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                SensorService.ACTION_STATUS_UPDATE -> {
                    val status = intent.getStringExtra(SensorService.EXTRA_STATUS) ?: return
                    updateStatus(status)
                }
                SensorService.ACTION_DR_UPDATE -> {
                    val posX = intent.getDoubleExtra(SensorService.EXTRA_POS_X, 0.0)
                    val posY = intent.getDoubleExtra(SensorService.EXTRA_POS_Y, 0.0)
                    val heading = intent.getDoubleExtra(SensorService.EXTRA_HEADING, 0.0)
                    val speed = intent.getDoubleExtra(SensorService.EXTRA_SPEED, 0.0)
                    val steps = intent.getIntExtra(SensorService.EXTRA_STEPS, 0)
                    updateDrDisplay(posX, posY, heading, speed, steps)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { results ->
            val allGranted = results.all { it.value }
            if (allGranted) {
                startStreamingSession()
                updateStatus("Permissions granted — streaming")
            } else {
                updateStatus("Permissions required to stream")
            }
        }

        setContentView(R.layout.activity_main)

        startButton = findViewById(R.id.startStreamingButton)
        stopButton = findViewById(R.id.stopStreamingButton)
        connectionStatusText = findViewById(R.id.connectionStatusTextView)
        positionXText = findViewById(R.id.positionXTextView)
        positionYText = findViewById(R.id.positionYTextView)
        headingText = findViewById(R.id.headingTextView)
        speedText = findViewById(R.id.speedTextView)
        stepCountText = findViewById(R.id.stepCountTextView)

        startButton.setOnClickListener {
            requestRequiredPermissions()
        }

        stopButton.setOnClickListener {
            stopService(Intent(this, SensorService::class.java))
            updateStatus("Streaming stopped")
        }

        updateStatus("Idle")
    }

    override fun onResume() {
        super.onResume()
        val filter = IntentFilter().apply {
            addAction(SensorService.ACTION_STATUS_UPDATE)
            addAction(SensorService.ACTION_DR_UPDATE)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(statusReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(statusReceiver, filter)
        }
    }

    override fun onPause() {
        super.onPause()
        unregisterReceiver(statusReceiver)
    }

    private fun requestRequiredPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missing.isEmpty()) {
            startStreamingSession()
            return
        }

        updateStatus("Requesting permissions...")
        permissionLauncher.launch(missing.toTypedArray())
    }

    private fun startStreamingSession() {
        val intent = Intent(this, SensorService::class.java)
        ContextCompat.startForegroundService(this, intent)
    }

    private fun updateStatus(message: String) {
        connectionStatusText.text = message
    }

    private fun updateDrDisplay(posX: Double, posY: Double, heading: Double, speed: Double, steps: Int) {
        positionXText.text = String.format("North: %.2f m", posX)
        positionYText.text = String.format("East:  %.2f m", posY)
        headingText.text = String.format("Heading: %.1f°", heading)
        speedText.text = String.format("Speed: %.2f m/s", speed)
        stepCountText.text = String.format("Steps: %d", steps)
    }
}
