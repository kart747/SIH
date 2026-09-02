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

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val allGranted = results.all { it.value }
        if (allGranted) {
            startStreamingSession()
            updateStatus("Permissions granted")
        } else {
            updateStatus("Permissions required to stream")
        }
    }

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val status = intent?.getStringExtra(SensorService.EXTRA_STATUS) ?: "Idle"
            updateStatus(status)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        startButton = findViewById(R.id.startStreamingButton)
        stopButton = findViewById(R.id.stopStreamingButton)
        connectionStatusText = findViewById(R.id.connectionStatusTextView)

        startButton.setOnClickListener {
            requestRequiredPermissions()
        }

        stopButton.setOnClickListener {
            stopService(Intent(this, SensorService::class.java))
            updateStatus("Streaming stopped")
        }

        updateStatus("Idle")
    }

    override fun onStart() {
        super.onStart()
        registerReceiver(statusReceiver, IntentFilter(SensorService.ACTION_STATUS_UPDATE))
    }

    override fun onStop() {
        super.onStop()
        unregisterReceiver(statusReceiver)
    }

    private fun requestRequiredPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            permissions.add(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
        }

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

        permissionLauncher.launch(missing.toTypedArray())
    }

    private fun startStreamingSession() {
        val intent = Intent(this, SensorService::class.java)
        ContextCompat.startForegroundService(this, intent)
    }

    private fun updateStatus(message: String) {
        connectionStatusText.text = message
    }
}
