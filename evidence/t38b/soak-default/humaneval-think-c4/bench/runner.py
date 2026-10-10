
import json, os, subprocess, sys
progs = sorted(os.listdir("/work/programs"))
timeout = float(sys.argv[1])
out = {}
for name in progs:
    path = os.path.join("/work/programs", name)
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        out[name[:-3]] = {"passed": p.returncode == 0, "returncode": p.returncode,
                          "stderr_tail": p.stderr[-600:]}
    except subprocess.TimeoutExpired:
        out[name[:-3]] = {"passed": False, "returncode": None, "stderr_tail": "timeout"}
with open("/out/results.json", "w") as f:
    json.dump(out, f, indent=1)
