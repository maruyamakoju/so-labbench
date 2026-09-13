param(
  [Parameter(Mandatory=$true)][string]$PolicyPath,
  [Parameter(Mandatory=$true)][string]$RepoPrefix,   # e.g. eval_c0_
  [int]$Trials = 30,
  [int]$EpisodeSec = 40,
  [string]$Cameras = '{fixed: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}',
  [string]$Task = "Pick the red-capped tube from rack A and place it into the marked slot of rack B"
)
# SO-LabBench trial orchestrator: homing -> single-episode policy run -> temp gate, repeated N times.
# Connect-failure retries and per-trial logging built in. Log: labbench_trials.log (ASCII).
$CFG = Get-Content "$(Split-Path $MyInvocation.MyCommand.Path)\labbench_config.json" -Raw | ConvertFrom-Json
$LE = Split-Path $CFG.paths.env_python
$HERE = Split-Path $MyInvocation.MyCommand.Path
$WORK = $CFG.paths.work_dir
$LOG = "$HERE\labbench_trials.log"
function TLog($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File $LOG -Append -Encoding ascii }

function Get-MaxTemp {
  $out = & "$LE\python.exe" "$HERE\health_motors.py" 2>$null
  $temps = $out | Select-String "temp=(\d+)C" -AllMatches | ForEach-Object { $_.Matches } | ForEach-Object { [int]$_.Groups[1].Value }
  if ($temps.Count -eq 0) { return -1 }
  return ($temps | Measure-Object -Maximum).Maximum
}

"" | Out-File $LOG -Encoding ascii
TLog "TRIALS START prefix=$RepoPrefix n=$Trials policy=$PolicyPath"

for ($i = 1; $i -le $Trials; $i++) {
  # temperature gate
  $t = Get-MaxTemp
  while ($t -ge 62) {
    TLog "trial $i : cooling (max temp ${t}C) - waiting 90s"
    Start-Sleep -Seconds 90
    $t = Get-MaxTemp
  }

  $ok = $false
  for ($try = 1; $try -le 3 -and -not $ok; $try++) {
    & "$LE\python.exe" "$HERE\home_pose.py" *>> "$HERE\labbench_home.log"
    Start-Sleep -Seconds 2
    $ds = Join-Path $CFG.paths.hf_cache "$RepoPrefix$i"
    if (Test-Path $ds) { Remove-Item $ds -Recurse -Force -Confirm:$false }

    TLog "trial $i try $try : recording"
    $argStr = "--robot.type=so101_follower --robot.port=$($CFG.robot.port) --robot.id=$($CFG.robot.id) " +
      "--robot.cameras=`"$Cameras`" --dataset.repo_id=maruo/$RepoPrefix$i " +
      "--dataset.single_task=`"$Task`" --policy.path=$PolicyPath " +
      "--dataset.num_episodes=1 --dataset.episode_time_s=$EpisodeSec --dataset.reset_time_s=0 " +
      "--dataset.push_to_hub=false --play_sounds=false"
    $p = Start-Process -FilePath "$LE\Scripts\lerobot-record.exe" -ArgumentList $argStr -WorkingDirectory $WORK -WindowStyle Hidden -RedirectStandardOutput "$HERE\trial_run.log" -RedirectStandardError "$HERE\trial_run.err" -PassThru -Wait
    $err = Get-Content "$HERE\trial_run.err" -Raw -ErrorAction SilentlyContinue
    if ($err -match "Missing motor") { TLog "trial $i try $try : missing motor, retry"; Start-Sleep -Seconds 5; continue }
    if (Test-Path "$ds\meta\info.json") { $ok = $true; TLog "trial $i : saved OK" }
    else { TLog "trial $i try $try : no data (exit=$($p.ExitCode))"; Start-Sleep -Seconds 3 }
  }
  if (-not $ok) { TLog "trial $i : FAILED after retries" }
  Start-Sleep -Seconds 8   # operator window: reset the object between trials
}
TLog "TRIALS DONE"
