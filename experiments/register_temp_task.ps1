# ELEVATED (one-time UAC "Yes" from the user): registers + starts a SYSTEM
# scheduled task that samples CPU thermal-zone temperature every 10 s.
$task = "ZCodeTempSensor"
$loop = "D:\Adaptive_Quantum_Gating\experiments\temp_sensor_loop.ps1"
schtasks /Create /TN $task /RU SYSTEM /RL HIGHEST /F /SC ONCE /ST 23:59 /TR "powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File $loop"
schtasks /Run /TN $task
"registered+run" | Out-File "D:\Adaptive_Quantum_Gating\results\logs\temp_task_registered.txt" -Encoding utf8
