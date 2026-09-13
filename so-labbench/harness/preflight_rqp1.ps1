# Delivery-day preflight for RQ-P1. Run BEFORE aligning cameras. Every check prints OK / FAIL
# with the reason; nothing here moves the arm.
#   $env:LABBENCH_CONFIG = "<...>\labbench_config_rqp1.json"; powershell -File preflight_rqp1.ps1
$HERE = Split-Path $MyInvocation.MyCommand.Path
$CFGPATH = if ($env:LABBENCH_CONFIG) { $env:LABBENCH_CONFIG } else { "$HERE\labbench_config_rqp1.json" }
$CFG = Get-Content $CFGPATH -Raw | ConvertFrom-Json
$PY = $CFG.paths.env_python
# The motor tools live in the recording environment. Naming the analysis environment
# through the default config rather than as a literal keeps this script - whose whole
# purpose is checking the bench is reproducible - free of a path to one machine.
$DEFAULT_CFG = Join-Path $HERE "labbench_config.json"
$OLDPY = if (Test-Path $DEFAULT_CFG) { (Get-Content $DEFAULT_CFG -Raw | ConvertFrom-Json).paths.env_python } else { $PY }
$fails = 0
function Check($name, $ok, $detail) { if ($ok) { Write-Host ("  OK   " + $name + "  " + $detail) -ForegroundColor Green } else { Write-Host ("  FAIL " + $name + "  " + $detail) -ForegroundColor Red; $script:fails++ } }

Write-Host "config: $CFGPATH"
Check "env python" (Test-Path $PY) $PY
$v = & $PY -c "import lerobot,torch;print(lerobot.__version__, torch.__version__, torch.cuda.is_available())" 2>$null
Check "lerobot import + CUDA" ($v -match "True") "$v"

# cameras: open each configured index at the configured size and count frames for 2 s
$camScript = @'
import cv2, time, json, sys
cams = json.loads(open(sys.argv[1], encoding="ascii").read())
for name, c in cams.items():
    cap = cv2.VideoCapture(int(c["index"]), cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"{name}: FAIL cannot open index {c['index']}"); continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["w"]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["h"]); cap.set(cv2.CAP_PROP_FPS, c["fps"])
    t = time.time(); n = 0; shape = None
    while time.time() - t < 2.0:
        ok, f = cap.read()
        if ok: n += 1; shape = f.shape
    fps = n / 2.0
    print(f"{name}: {'OK' if fps >= c['fps']*0.8 else 'FAIL'} index {c['index']} got {shape[1]}x{shape[0]} at {fps:.1f} fps (want {c['w']}x{c['h']} at {c['fps']})" if shape else f"{name}: FAIL no frames")
    cap.release()
'@
$camSpec = @{}
foreach ($p in $CFG.cameras.PSObject.Properties) { if ($p.Name.StartsWith("_")) { continue }; $c = $p.Value; if ($c.PSObject.Properties.Name -contains "enabled" -and -not $c.enabled) { continue }
  $camSpec[$p.Name] = @{ index = $c.index_or_path; w = $c.width; h = $c.height; fps = $c.fps } }
$tmp = Join-Path $env:TEMP "preflight_cams.py"; $camScript | Out-File $tmp -Encoding ascii
$spec = Join-Path $env:TEMP "preflight_cams.json"; ($camSpec | ConvertTo-Json -Compress) | Out-File $spec -Encoding ascii
$camOut = @(& $PY $tmp $spec 2>&1 | Where-Object { $_ -match "^(front|top|wrist|\w+): " })
if ($camOut.Count -eq 0) { Check "camera probe" $false "probe produced no output (python error?)" }
foreach ($line in $camOut) { Check ("camera " + ($line -split ":")[0]) ($line -match ": OK") $line }

# calibration in the new layout
$cal = "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\robots\so_follower\$($CFG.robot.id).json"
Check "follower calibration (new layout)" (Test-Path $cal) $cal

# motors answer on the port (old env tool, read-only)
$m = & $OLDPY "$HERE\health_motors.py" 2>$null
$nOK = ($m | Select-String "OK").Count
Check "motors respond" ($nOK -eq 6) "$nOK/6 motors OK on $($CFG.robot.port) (power on? USB in?)"

# policies cached
foreach ($p in $CFG.policies.PSObject.Properties) { if ($p.Name.StartsWith("_")) { continue }
  $id = $p.Value; $dir = "$env:USERPROFILE\.cache\huggingface\hub\models--" + ($id -replace "/", "--")
  Check ("policy cached: " + $p.Name) (Test-Path $dir) $id }

# reference frames for alignment
foreach ($cam in @("front", "top", "wrist")) { $r = Join-Path $HERE ($CFG.alignment.reference_dir + "\" + $CFG.alignment.$cam); Check ("reference frame " + $cam) (Test-Path $r) $r }

# disk
$free = [math]::Round((Get-PSDrive C).Free / 1GB); Check "disk free >= 20 GB" ($free -ge 20) "$free GB"

Write-Host ""
if ($fails -eq 0) { Write-Host "PREFLIGHT: GO" -ForegroundColor Green } else { Write-Host "PREFLIGHT: $fails problem(s) - fix before aligning" -ForegroundColor Red }
exit $fails
