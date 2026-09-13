# Post-processing for one condition: score, then record the covariates that let a
# surprising result be explained later rather than argued about.
#   powershell -File finish_condition.ps1 -Condition pilot -Pairs 5
# Produces per-model scores, the true control rate of each trial, the camera residual
# of each trial, and a filmstrip to label from.
param(
  [string]$Condition = "pilot",
  [int]$Pairs = 5,
  [string[]]$Models = @("act", "smolvla"),
  [string]$Study = "rq1"          # repo ids are eval_<Study>_<Condition>_<model>_<n>
)
$Models = @($Models | ForEach-Object { $_ -split "," } | Where-Object { $_ })   # -File passes "a,b,c" as one string
$HERE = Split-Path $MyInvocation.MyCommand.Path
$CFGPATH = if ($env:LABBENCH_CONFIG) { $env:LABBENCH_CONFIG } else { "$HERE\labbench_config.json" }
$PY = (Get-Content $CFGPATH -Raw | ConvertFrom-Json).paths.env_python

Push-Location $HERE
try {
  foreach ($m in $Models) {
    $prefix = "eval_${Study}_${Condition}_${m}_"
    Write-Host "`n=== score $prefix ===" -ForegroundColor Cyan
    & $PY score_rq1.py $prefix $Pairs
    if ($LASTEXITCODE -ne 0) { Write-Host "  score_rq1.py exited $LASTEXITCODE" -ForegroundColor Yellow }
    Write-Host "`n=== filmstrip $prefix ===" -ForegroundColor Cyan
    & $PY make_filmstrip.py $prefix $Pairs
  }

  # --study/--condition matters: with no filter these three walk the whole manifest, so
  # finishing one condition would rewrite the rate, residual and closure record of every
  # other study with a different set of trials.
  $scope = @("--study", $Study, "--condition", $Condition)

  Write-Host "`n=== control rate per trial ===" -ForegroundColor Cyan
  & $PY trial_timing.py @scope
  if ($LASTEXITCODE -ne 0) { Write-Host "  trial_timing.py exited $LASTEXITCODE" -ForegroundColor Yellow }

  Write-Host "`n=== camera residual per trial ===" -ForegroundColor Cyan
  & $PY alignment_log.py @scope
  if ($LASTEXITCODE -ne 0) { Write-Host "  alignment_log.py exited $LASTEXITCODE" -ForegroundColor Yellow }

  Write-Host "`n=== commanded vs achieved closure ===" -ForegroundColor Cyan
  & $PY command_vs_achieved.py @scope
  if ($LASTEXITCODE -ne 0) { Write-Host "  command_vs_achieved.py exited $LASTEXITCODE" -ForegroundColor Yellow }

  if ($Study -eq "rqp1") {
    Write-Host "`n=== RQ-P1 human-judged success vs ArmnetBench (needs rqp1_human_labels.md filled in) ===" -ForegroundColor Cyan
    & $PY rqp1_results.py $Condition
  }

  if ($Condition -eq "pilot") {
    Write-Host "`n=== agreement (needs pilot_human_labels.md filled in) ===" -ForegroundColor Cyan
    & $PY pilot_agreement.py
  }
}
finally { Pop-Location }
