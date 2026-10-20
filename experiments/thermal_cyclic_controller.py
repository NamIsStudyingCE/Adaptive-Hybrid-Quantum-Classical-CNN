import subprocess
import time
import os

PIDS = [13768, 21144, 8888, 25136]

def set_hardware_mode(mode="boost"):
    # mode "boost": 12 cores (0xFFF), Normal Priority
    # mode "cool": 4 cores (0x0F), BelowNormal Priority
    if mode == "boost":
        affinity = 0xFFF
        priority = "Normal"
    else:
        affinity = 0x0F
        priority = "BelowNormal"
        
    ps_cmd = f"""
    $pids = 13768, 21144, 8888, 25136
    foreach ($id in $pids) {{
        try {{
            $p = Get-Process -Id $id -ErrorAction SilentlyContinue
            if ($p) {{
                $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::{priority}
                $p.ProcessorAffinity = {affinity}
            }}
        }} catch {{}}
    }}
    """
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    print(f"[{time.strftime('%H:%M:%S')}] Switched hardware mode to: {mode.upper()} (Affinity: {hex(affinity)}, Priority: {priority})")

def thermal_cycle_daemon():
    print("=" * 70)
    print("CYCLIC THERMAL CONTROLLER ACTIVE: 15 mins BOOST <-> 10 mins COOL")
    print("=" * 70)
    # Start with cool mode since CPU was warm, then switch to boost
    while True:
        # Cool down phase (10 minutes = 600s)
        set_hardware_mode("cool")
        print(f"[{time.strftime('%H:%M:%S')}] Active COOLING phase for 10 minutes...")
        time.sleep(600)
        
        # Max performance boost phase (15 minutes = 900s)
        set_hardware_mode("boost")
        print(f"[{time.strftime('%H:%M:%S')}] Active MAXIMUM PERFORMANCE phase for 15 minutes...")
        time.sleep(900)

if __name__ == "__main__":
    thermal_cycle_daemon()
