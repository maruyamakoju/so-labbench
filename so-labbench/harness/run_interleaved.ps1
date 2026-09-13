param(
  [string]$Condition = "pilot",
  [int]$Pairs = 5,          # rounds; each round runs every arm once -> Arms.Count*Pairs trials
  [int]$EpisodeSec = 45,
  [int]$ResetSec = 0,          # 0 = take task.reset_time_s from the config
  [string[]]$Arms = @("act", "smolvla"),   # names under "policies" in labbench_config.json
  [string]$Study = "rq1",                  # repo ids are eval_<Study>_<Condition>_<arm>_<round>
  [switch]$DryRun                          # print the schedule and the record command; touch nothing
)
# Interleaved trial runner (RQ1 preregistration item 2, generalised to N arms for RQ0):
#   round k: the arms rotated by (k-1) mod N, position P((k-1)%5+1)
#   For two arms this is exactly the old pair rule (k odd: ACT->SmolVLA, k even: SmolVLA->ACT).
#   Writes one manifest row per trial to rq1_manifest.csv (model column = arm name), which is
#   what score_rq1 / trial_timing / alignment_log / command_vs_achieved read.
# Machine-specific values live in labbench_config.json so a third party edits one file.
$Arms = @($Arms | ForEach-Object { $_ -split "," } | Where-Object { $_ })   # -File passes "a,b,c" as one string
$HERE = Split-Path $MyInvocation.MyCommand.Path
# LABBENCH_CONFIG selects the machine/study config (labbench.py honours the same variable),
# so the RQ-P1 setup (other env, three cameras, 20 fps, hub policies) is one file, not a fork.
$CFGPATH = if ($env:LABBENCH_CONFIG) { $env:LABBENCH_CONFIG } else { "$HERE\labbench_config.json" }
$CFG = Get-Content $CFGPATH -Raw | ConvertFrom-Json
$LE = Split-Path $CFG.paths.env_python
$WORK = $CFG.paths.work_dir
$HF = $CFG.paths.hf_cache
$PORT = $CFG.robot.port
$ROBOT_ID = $CFG.robot.id
$FPS = if ($CFG.dataset -and $CFG.dataset.fps) { [int]$CFG.dataset.fps } else { 30 }
$MAX_RETRIES = if ($CFG.safety -and $CFG.safety.max_connect_retries) { [int]$CFG.safety.max_connect_retries } else { 3 }
$TEMP_PAUSE_C = if ($CFG.safety -and $CFG.safety.temp_pause_threshold_c) { [int]$CFG.safety.temp_pause_threshold_c } else { 62 }
$TEMP_RESUME_C = if ($CFG.safety -and $CFG.safety.temp_resume_threshold_c) { [int]$CFG.safety.temp_resume_threshold_c } else { 57 }
if ($ResetSec -le 0) { $ResetSec = if ($CFG.task -and $CFG.task.reset_time_s) { [int]$CFG.task.reset_time_s } else { 20 } }
$TASK = if ($CFG.task -and $CFG.task.instruction) { $CFG.task.instruction } else { "Pick up the remote control" }
# every enabled camera in the config becomes one entry of --robot.cameras, in config order
$CAMS = @()
foreach ($prop in $CFG.cameras.PSObject.Properties) {
  if ($prop.Name.StartsWith("_")) { continue }          # "_note" and friends are documentation, not cameras
  $c = $prop.Value
  if ($c.PSObject.Properties.Name -contains "enabled" -and -not $c.enabled) { continue }
  $extra = if ($c.PSObject.Properties.Name -contains "fourcc" -and $c.fourcc) { ", fourcc: $($c.fourcc)" } else { "" }
  $CAMS += "$($prop.Name): {type: $($c.type), index_or_path: $($c.index_or_path), width: $($c.width), height: $($c.height), fps: $($c.fps)$extra}"
}
$CAMERA_DICT = "{" + ($CAMS -join ", ") + "}"
$FIRST_CAM = ($CFG.cameras.PSObject.Properties | Where-Object { -not $_.Name.StartsWith("_") -and (-not ($_.Value.PSObject.Properties.Name -contains "enabled") -or $_.Value.enabled) } | Select-Object -First 1).Name
$LOG = "$HERE\rq1_run.log"
$MANIFEST = "$HERE\rq1_manifest.csv"
$POLICIES = @{}
foreach ($a in $Arms) {
  $path = $CFG.policies.$a
  if (-not $path) { Write-Host "arm '$a' has no entry under policies in labbench_config.json"; exit 1 }
  $isLocal = ($path -match ":" ) -or ($path -match "^[\/]")     # a hub id like user/model has neither
  if (-not $DryRun -and $isLocal -and -not (Test-Path "$path\model.safetensors")) { Write-Host "arm '$a': no model at $path"; exit 1 }
  $POLICIES[$a] = $path
}

function TLog($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding ascii }
if (-not $DryRun -and -not (Test-Path $MANIFEST)) {
  "run_id,timestamp,model,condition,start_position,repo_id,episode_sec,video_path" | Out-File $MANIFEST -Encoding ascii
}

function Get-MaxTemp {
  $out = & "$LE\python.exe" "$HERE\health_motors.py" 2>$null
  $temps = $out | Select-String "temp=(\d+)C" -AllMatches | ForEach-Object { $_.Matches } | ForEach-Object { [int]$_.Groups[1].Value }
  if ($temps.Count -eq 0) { return -1 }
  return ($temps | Measure-Object -Maximum).Maximum
}

# The control rate is not a constant of this rig - it fell from 23.8 to 14.9 Hz inside
# one pilot session, which silently changes what every frame-counted threshold means.
# Record the machine's state next to each trial so a future drop can be attributed
# instead of guessed at. Never fatal: a missing sample must not cost a trial.
$SYSLOG = "$HERE\rq1_system.csv"
if (-not $DryRun -and -not (Test-Path $SYSLOG)) {
  "run_id,timestamp,cpu_busy_pct,free_mem_mb,gpu_temp_c,gpu_clock_mhz,motor_max_temp_c" | Out-File $SYSLOG -Encoding ascii
}
function Log-System([string]$RunId, [int]$MotorTemp) {
  try {
    $cpu = [math]::Round((Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction Stop).CounterSamples[0].CookedValue, 1)
  } catch { $cpu = -1 }
  try {
    $mem = [math]::Round((Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).FreePhysicalMemory / 1KB, 0)
  } catch { $mem = -1 }
  $gt = -1; $gc = -1
  try {
    $g = (& nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader,nounits 2>$null) -split ','
    if ($g.Count -ge 2) { $gt = [int]$g[0].Trim(); $gc = [int]$g[1].Trim() }
  } catch { }
  "$RunId,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),$cpu,$mem,$gt,$gc,$MotorTemp" | Out-File $SYSLOG -Append -Encoding ascii
}

function Run-Trial([string]$Model, [string]$Policy, [int]$Index, [string]$Pos) {
  $repo = "eval_${Study}_${Condition}_${Model}_$Index"
  if ($DryRun) {
    Write-Host "  $repo  pos=$Pos  policy=$Policy"
    if ($Index -eq 1 -and -not $script:shownArgs) { $script:shownArgs = $true
      Write-Host "  record args (first trial): --robot.cameras=$CAMERA_DICT --dataset.fps=$FPS --dataset.single_task=`"$TASK`" env=$LE" }
    return
  }
  $ok = $false
  for ($try = 1; $try -le $MAX_RETRIES -and -not $ok; $try++) {
    $t = Get-MaxTemp
    # A thermometer that cannot be read is not a cool motor. health_motors.py prints
    # temp=UNREADABLE in that case, Get-MaxTemp returns -1, and stopping is the safe reading.
    if ($t -lt 0) { TLog "$repo : motor temperature unreadable - stopping rather than assuming cool"; return }
    while ($t -ge $TEMP_PAUSE_C) {
      TLog "$repo : cooling (${t}C, resume below ${TEMP_RESUME_C}C)"
      Start-Sleep -Seconds 90
      $t = Get-MaxTemp
      if ($t -lt 0) { TLog "$repo : temperature unreadable while cooling - stopping"; return }
    }
    & "$LE\python.exe" "$HERE\home_pose.py" *>> "$HERE\rq1_home.log"
    # home_pose exits nonzero when a joint did not reach the centre. Recording anyway would
    # start the trial from a pose nobody chose, so retry from the top instead.
    if ($LASTEXITCODE -ne 0) { TLog "$repo : home pose incomplete (see rq1_home.log), retrying"; Start-Sleep -Seconds 3; continue }
    Start-Sleep -Seconds 2
    $ds = Join-Path $HF $repo
    if (Test-Path $ds) { Remove-Item $ds -Recurse -Force -Confirm:$false }
    Log-System $repo $t
    TLog "$repo pos=$Pos try $try : recording"
    $argStr = "--robot.type=so101_follower --robot.port=$PORT --robot.id=$ROBOT_ID " +
      "--robot.cameras=`"$CAMERA_DICT`" " +
      "--dataset.repo_id=maruo/$repo --dataset.fps=$FPS " + "--dataset.single_task=`"$TASK`" " +
      "--policy.path=$Policy --dataset.num_episodes=1 --dataset.episode_time_s=$EpisodeSec --dataset.reset_time_s=0 " +
      "--dataset.push_to_hub=false --play_sounds=false"
    $p = Start-Process -FilePath "$LE\Scripts\lerobot-record.exe" -ArgumentList $argStr -WorkingDirectory $WORK -WindowStyle Hidden -RedirectStandardOutput "$HERE\rq1_trial.log" -RedirectStandardError "$HERE\rq1_trial.err" -PassThru -Wait
    $err = Get-Content "$HERE\rq1_trial.err" -Raw -ErrorAction SilentlyContinue
    if ($err -match "Missing motor") { TLog "$repo : missing motor, retry"; Start-Sleep -Seconds 5; continue }
    if (Test-Path "$ds\meta\info.json") {
      $ok = $true
      # Ask labbench.py rather than rebuilding the layout here: it reads meta/info.json and
      # knows v2.1 from v3.0. A second implementation in a second language is how the two drift apart.
      $video = & "$LE\python.exe" "$HERE\labbench.py" --episode-video "$repo" $FIRST_CAM 2>$null
      if (-not $video) { $video = "" }
      "$repo,$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'),$Model,$Condition,$Pos,maruo/$repo,$EpisodeSec,$video" | Out-File $MANIFEST -Append -Encoding ascii
      TLog "$repo : saved OK"
    } else { TLog "$repo : no data (exit=$($p.ExitCode))"; Start-Sleep -Seconds 3 }
  }
  if (-not $ok) { TLog "$repo : FAILED after retries" }
  Start-Sleep -Seconds $ResetSec
}

if ($DryRun) { function TLog($m) { Write-Host $m } } else { "" | Out-File $LOG -Encoding ascii }
TLog "$($Study.ToUpper()) RUN START condition=$Condition rounds=$Pairs arms=$($Arms -join ',')"
$N = $Arms.Count
for ($k = 1; $k -le $Pairs; $k++) {
  $pos = "P" + ((($k - 1) % 5) + 1)
  $shift = ($k - 1) % $N
  $order = @(); for ($j = 0; $j -lt $N; $j++) { $order += $Arms[($j + $shift) % $N] }
  TLog "ROUND $k position=$pos order=$($order -join '>')"
  foreach ($arm in $order) { Run-Trial $arm $POLICIES[$arm] $k $pos }
}
TLog "$($Study.ToUpper()) RUN DONE condition=$Condition"
