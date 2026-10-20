# Thermal controller v3 — temp-gated with process suspension.
# Cycle: 10 min FULL (all workers resumed, full affinity) ->
#        15 min WINDOW in which CPU/package temp must stay <= 75 C.
# WINDOW enforcement: escalate suspension of worker processes
#   level 0: all running      (affinity: every-other logical CPU)
#   level 1: 2 of 4 suspended
#   level 2: 1 running
#   level 3: all suspended (hard pause until temp <= 72 C)
# Suspension (NtSuspendProcess) is non-destructive: worker state freezes and
# resumes exactly where it stopped. All suspended processes are always resumed
# at the start of the next FULL phase and when the controller exits.
$ErrorActionPreference = 'SilentlyContinue'
$log = "results/logs/thermal_duty.log"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ProcCtl {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
    [DllImport("ntdll.dll")] public static extern uint NtSuspendProcess(IntPtr h);
    [DllImport("ntdll.dll")] public static extern uint NtResumeProcess(IntPtr h);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
"@
$PROCESS_SUSPEND_RESUME = 0x0800

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
function Suspend-Worker($wpid) {
    try {
        $h = [ProcCtl]::OpenProcess($PROCESS_SUSPEND_RESUME, $false, [int]$wpid)
        if ($h -ne [IntPtr]::Zero) { [void][ProcCtl]::NtSuspendProcess($h); [ProcCtl]::CloseHandle($h) | Out-Null; return $true }
    } catch { }
    return $false
}
function Resume-Worker($wpid) {
    try {
        $h = [ProcCtl]::OpenProcess($PROCESS_SUSPEND_RESUME, $false, [int]$wpid)
        if ($h -ne [IntPtr]::Zero) { [void][ProcCtl]::NtResumeProcess($h); [ProcCtl]::CloseHandle($h) | Out-Null }
    } catch { }
}
function Resume-All($suspended) {
    foreach ($wpid in @($suspended)) { Resume-Worker $wpid }
    $suspended.Clear()
}

$script:suspended = New-Object System.Collections.ArrayList
$logical = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$halfMask = 0
for ($i = 0; $i -lt $logical; $i += 2) { $halfMask = $halfMask -bor [long][Math]::Pow(2, $i) }
$fullMask = 0
for ($i = 0; $i -lt $logical; $i++) { $fullMask = $fullMask -bor [long][Math]::Pow(2, $i) }
Log "controller v3 start: suspension-based temp cap 75C (escalate 0/2/3/4 frozen), FULL 10 min per cycle"

$cycle = 0
try {
    while ($true) {
        $cycle++
        $pids = @(Get-WorkerPids)
        if ($pids.Count -eq 0) { break }
        Resume-All $script:suspended
        foreach ($wpid in $pids) {
            try { (Get-Process -Id $wpid -ErrorAction Stop).ProcessorAffinity = [IntPtr]$fullMask } catch { }
        }
        Log ("cycle {0}: FULL phase (10 min) - all {1} workers running" -f $cycle, $pids.Count)
        Start-Sleep -Seconds 600

        Log ("cycle {0}: WINDOW phase (15 min, cap 75C)" -f $cycle)
        $level = 0
        $t0 = Get-Date
        while (((Get-Date) - $t0).TotalSeconds -lt 900) {
            Start-Sleep -Seconds 15
            $pids = @(Get-WorkerPids)
            if ($pids.Count -eq 0) { break }
            $t = Read-Temp
            if ($null -eq $t) { continue }
            $n = $pids.Count
            $desired = 0
            if ($t -gt 80) { $desired = [Math]::Min(3, [Math]::Max(0, $n - 1)) }
            elseif ($t -gt 75) { $desired = [Math]::Min(2, [Math]::Max(0, [int][Math]::Floor($n / 2))) }
            elseif ($t -le 72) { $desired = 0 }
            else { $desired = $level }
            if ($desired -ne $level) {
                $running = @($pids | Where-Object { -not $script:suspended.Contains([int]$_) })
                $toSuspend = $desired
                $toResume  = $n - $desired
                if ($toSuspend -lt $level -or $desired -lt $level) {
                    # de-escalating: resume first
                    foreach ($wpid in @($script:suspended)) {
                        if ($toResume -le 0) { break }
                        Resume-Worker $wpid
                        $script:suspended.Remove([int]$wpid)
                        $toResume--
                    }
                }
                $running = @($pids | Where-Object { -not $script:suspended.Contains([int]$_) })
                while ($script:suspended.Count -lt $desired -and $running.Count -gt 0) {
                    $victim = $running[-1]
                    if (Suspend-Worker $victim) {
                        $script:suspended.Add([int]$victim) | Out-Null
                        Log ("  temp {0:N1}C -> suspend {1} (level {2}/3)" -f $t, $victim, $script:suspended.Count)
                    }
                    $running = @($running | Select-Object -First ($running.Count - 1))
                }
                $level = $desired
            }
        }
        if (@(Get-WorkerPids).Count -eq 0) { break }
    }
}
finally {
    Log "controller v3 exiting - resuming ALL suspended workers"
    Resume-All $script:suspended
}
