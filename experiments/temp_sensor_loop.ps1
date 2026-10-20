# Runs elevated (as SYSTEM via scheduled task). Writes CPU thermal-zone temp
# (Celsius) to results/logs/cpu_temp.txt every 10 seconds.
$log = "D:\Adaptive_Quantum_Gating\results\logs\cpu_temp.txt"
while ($true) {
    try {
        $max = -999.0
        Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop |
            ForEach-Object {
                $c = $_.CurrentTemperature / 10.0 - 273.15
                if ($c -gt $max) { $max = $c }
            }
        if ($max -gt -100) {
            Add-Content -Path $log -Value ("{0} {1:N1}" -f (Get-Date -Format 'HH:mm:ss'), $max)
        } else {
            Add-Content -Path $log -Value ("{0} ERR" -f (Get-Date -Format 'HH:mm:ss'))
        }
    } catch {
        Add-Content -Path $log -Value ("{0} ERR {1}" -f (Get-Date -Format 'HH:mm:ss'), $_.Exception.Message)
    }
    Start-Sleep -Seconds 10
}
