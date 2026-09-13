param(
  [string[]]$Arms = @("23complete", "23random"),
  [int]$Steps = 100000,
  [int]$SaveFreq = 20000,
  [string]$Tag = ""          # suffix for the output dir; use e.g. -Tag _smoke -Steps 10 to test the batch
)
# RQ0 training batch: retrain ACT on episode subsets of the same 60-demo dataset.
#   powershell -File train_rq0.ps1                      # both arms, sequential, one night
#   powershell -File train_rq0.ps1 -Arms 23complete     # one arm
#
# Every setting other than the episode list is copied from the frozen act_aug_v2_60ep
# checkpoint's train_config.json (100k steps, batch 8, seed 1000, image transforms on,
# lr 1e-5) so that the episode list is the only thing that differs between arms.
# The arms themselves come from rq0_subsets.json (written by rq0_subsets.py), never typed here.
# Checkpoints every 20k are kept: 100k*23/60 ~ 38k, so checkpoint 040000 is the
# matched-epoch reading and checkpoint last is the matched-steps reading.
$Arms = @($Arms | ForEach-Object { $_ -split "," } | Where-Object { $_ })   # -File passes "a,b,c" as one string
$HERE = Split-Path $MyInvocation.MyCommand.Path
$CFG = Get-Content "$HERE\labbench_config.json" -Raw | ConvertFrom-Json
$TRAIN = Join-Path $CFG.paths.lerobot_scripts "lerobot-train.exe"
$WORK = $CFG.paths.work_dir
$SUBSETS = Get-Content "$HERE\rq0_subsets.json" -Raw | ConvertFrom-Json
$LOG = "$HERE\rq0_train.log"
function TLog($m) { $line = "$(Get-Date -Format 'MM-dd HH:mm:ss') $m"; $line | Out-File $LOG -Append -Encoding ascii; Write-Host $line }

Set-Location $WORK
foreach ($arm in $Arms) {
  $spec = $SUBSETS.arms.$arm
  if (-not $spec) { TLog "unknown arm '$arm' (rq0_subsets.json has: $($SUBSETS.arms.PSObject.Properties.Name -join ', '))"; exit 1 }
  $eps = "[" + ($spec.episodes -join ",") + "]"
  $out = "outputs\train\rq0_act_$arm$Tag"
  if (Test-Path "$out\checkpoints\last\pretrained_model\model.safetensors") { TLog "$arm already trained ($out), skipping"; continue }
  if (Test-Path $out) { TLog "$out exists without a final checkpoint - move it away or resume by hand"; exit 1 }
  TLog "arm=$arm n=$($spec.episodes.Count) episodes=$eps steps=$Steps -> $out"
  $t0 = Get-Date
  & $TRAIN `
    --dataset.repo_id=maruo/so101_pick_remote `
    --dataset.episodes=$eps `
    --dataset.image_transforms.enable=true `
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
    --batch_size=8 --steps=$Steps --save_freq=$SaveFreq --log_freq=200 --num_workers=4 --seed=1000 `
    --wandb.enable=false --output_dir=$out 2>&1 | Tee-Object -FilePath "$HERE\rq0_train_$arm.out" | Select-String "step:|Error|error" 
  $rc = $LASTEXITCODE
  TLog "arm=$arm finished rc=$rc elapsed=$([int]((Get-Date) - $t0).TotalMinutes) min"
  if ($rc -ne 0) { exit $rc }
}
TLog "RQ0 training batch done"
