# Thermal duty-cycle controller for benchmark_runner workers.
# Cycle: 10 min full CPU affinity -> 15 min pinned to every-other logical
# processor (~50% load, 1 thread per physical core) -> repeat.
# Self-terminates when no benchmark_runner python process remains; affinity is
# per-process and non-destructive (workers never crash from affinity changes).
$ErrorActionPreference = 'SilentlyContinue'
$log = "results/logs/thermal_duty.log"

function Log($msg) {
    $line = ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg)
    Add-Content -Path $log -Value $line
}

function Get-WorkerPids {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'benchmark_runner' } |
        Select-Object -ExpandProperty ProcessId
}

$logical = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$fullMask = 0
for ($i = 0; $i -lt $logical; $i++) { $fullMask = $fullMask -bor [long][Math]::Pow(2, $i) }
# half mask: alternate logical processors (1 thread per physical core on SMT CPUs)
$halfMask = 0
for ($i = 0; $i -lt $logical; $i += 2) { $halfMask = $halfMask -bor [long][Math]::Pow(2, $i) }

Log "controller start: logical=$logical full=0x$('{0:X}' -f $fullMask) half=0x$('{0:X}' -f $halfMask)"

function Set-AffinityFor($pids, $mask) {
    foreach ($wpid in $pids) {
        try {
            $p = Get-Process -Id $wpid -ErrorAction Stop
            $p.ProcessorAffinity = [IntPtr]$mask
        } catch { }
    }
}

$cycle = 0
while ($true) {
    $cycle++
    $pids = @(Get-WorkerPids)
    if ($pids.Count -eq 0) { break }
    Set-AffinityFor $pids $fullMask
    Log ("cycle {0}: FULL phase (10 min), workers: {1}" -f $cycle, ($pids -join ','))
    Start-Sleep -Seconds 600

    $pids = @(Get-WorkerPids)
    if ($pids.Count -eq 0) { break }
    Set-AffinityFor $pids $halfMask
    Log ("cycle {0}: HALF(50%) phase (15 min)" -f $cycle)
    Start-Sleep -Seconds 900
}

Log "no workers left - controller exits (processes ended on their own; nothing to restore)"
