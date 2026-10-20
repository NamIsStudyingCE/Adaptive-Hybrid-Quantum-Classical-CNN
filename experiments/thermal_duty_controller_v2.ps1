# Thermal duty-cycle controller v2.
# FULL phase: 10 min unrestricted (as before).
# WINDOW phase: 15 min in which CPU temperature must stay <= 75 C. Starts at
# ~50% affinity (every other logical CPU) and steps down to quarter / minimal
# mask while temp > 75, stepping back up with hysteresis (<= 72 C).
# Requires results/logs/cpu_temp.txt (elevated temp sensor task). If the file
# is absent/stale, falls back to the fixed 50% scheme and logs a warning.
# Self-terminates when no benchmark_runner workers remain.
$ErrorActionPreference = 'SilentlyContinue'
$log = "results/logs/thermal_duty.log"

function Log($msg) {
    Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg)
}
function Get-WorkerPids {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'benchmark_runner' } |
        Select-Object -ExpandProperty ProcessId
}
function Read-Temp {
    $line = Get-Content "results/logs/cpu_temp.txt" -Tail 1 -ErrorAction SilentlyContinue
    if ($line -match '^(\d\d:\d\d:\d\d) ([0-9.]+)$') {
        $age = ((Get-Date) - [datetime]::ParseExact($matches[1], 'HH:mm:ss', $null)).TotalSeconds
        if ($age -lt 90) { return [double]$matches[2] }
    }
    return $null
}
function Set-Level($pids, $level) {
    # 0 = half (every 2nd logical), 1 = quarter, 2 = minimal (1 thread)
    $masks = @()
    $n = 12
    $m0 = 0; for ($i = 0; $i -lt $n; $i += 2) { $m0 = $m0 -bor [long][Math]::Pow(2, $i) }
    $m1 = 0; for ($i = 0; $i -lt $n; $i += 4) { $m1 = $m1 -bor [long][Math]::Pow(2, $i) }
    $m2 = 1
    $masks = @($m0, $m1, $m2)
    foreach ($wpid in $pids) {
        try { (Get-Process -Id $wpid -ErrorAction Stop).ProcessorAffinity = [IntPtr]$masks[$level] } catch { }
    }
}
$names = @('HALF(~50%)', 'QUARTER(~25%)', 'MINIMAL(1 thread)')

$logical = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$fullMask = 0
for ($i = 0; $i -lt $logical; $i++) { $fullMask = $fullMask -bor [long][Math]::Pow(2, $i) }
Log "controller v2 start: logical=$logical temp-cap=75C window=15min"

function Set-AffinityFor($pids, $mask) {
    foreach ($wpid in $pids) {
        try { (Get-Process -Id $wpid -ErrorAction Stop).ProcessorAffinity = [IntPtr]$mask } catch { }
    }
}

$cycle = 0
while ($true) {
    $cycle++
    $pids = @(Get-WorkerPids)
    if ($pids.Count -eq 0) { break }
    Set-AffinityFor $pids $fullMask
    Log ("cycle {0}: FULL phase (10 min)" -f $cycle)
    Start-Sleep -Seconds 600

    $pids = @(Get-WorkerPids)
    if ($pids.Count -eq 0) { break }
    Log ("cycle {0}: WINDOW phase (15 min, cap 75C)" -f $cycle)
    $level = 0
    $warned = $false
    Set-Level $pids $level
    $t0 = Get-Date
    while (((Get-Date) - $t0).TotalSeconds -lt 900) {
        Start-Sleep -Seconds 15
        $pids = @(Get-WorkerPids)
        if ($pids.Count -eq 0) { break }
        $t = Read-Temp
        if ($null -eq $t) {
            if (-not $warned) { Log "WARN: no fresh cpu_temp.txt - fixed 50% fallback (75C cap NOT enforced)"; $warned = $true }
            Set-Level $pids 0
            continue
        }
        if ($t -gt 75 -and $level -lt 2) {
            $level++
            Set-Level $pids $level
            Log ("  temp {0:N1}C > 75 -> tighten to {1}" -f $t, $names[$level])
        } elseif ($t -le 72 -and $level -gt 0) {
            $level--
            Set-Level $pids $level
            Log ("  temp {0:N1}C <= 72 -> relax to {1}" -f $t, $names[$level])
        }
    }
    if (@(Get-WorkerPids).Count -eq 0) { break }
}
Log "no workers left - controller v2 exits"
