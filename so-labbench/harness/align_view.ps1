# Viewer window for align_camera.py's live preview file.
# The lerobot env has headless opencv, so the preview is written to disk and displayed here.
# Closing the window writes align_stop.flag, which stops the capture loop.
#
# The title bar says how old the frame on screen is. That is not decoration: the capture
# loop can stop - a camera unplugged, a permission error, the process killed - and this
# window would go on showing the last good frame indefinitely. Aiming a camera against a
# frozen preview produces a confident, wrong alignment, and nothing else would say so.
param(
  [string]$Image = "$PSScriptRoot\_frames\camera_live.png"
)
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

$stopFlag = Join-Path (Split-Path $Image) "align_stop.flag"
$STALE_AFTER_S = 3          # align_camera.py writes about every 0.15 s

$form = New-Object System.Windows.Forms.Form
$form.Text = "camera alignment - waiting for the first frame"
$form.Size = New-Object System.Drawing.Size(1000, 800)
$form.StartPosition = "CenterScreen"
$form.TopMost = $true

$box = New-Object System.Windows.Forms.PictureBox
$box.Dock = "Fill"
$box.SizeMode = "Zoom"
$form.Controls.Add($box)

$script:lastError = ""
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 200
$timer.Add_Tick({
  if (-not (Test-Path $Image)) {
    $form.Text = "camera alignment - no preview file yet ($Image)"
    return
  }
  $age = ((Get-Date) - (Get-Item $Image).LastWriteTime).TotalSeconds
  try {
    $bytes = [System.IO.File]::ReadAllBytes($Image)
    $ms = New-Object System.IO.MemoryStream(, $bytes)
    try {
      $img = [System.Drawing.Image]::FromStream($ms)
      if ($box.Image) { $box.Image.Dispose() }
      $box.Image = $img
    } finally {
      # The Image keeps its own copy once loaded; the stream would otherwise accumulate
      # five times a second for as long as the window is open.
      $ms.Dispose()
    }
    $script:lastError = ""
  } catch {
    # A half-written file is normal at this rate and fixes itself on the next tick. An
    # error that persists is not, so it goes in the title rather than being swallowed.
    $script:lastError = $_.Exception.Message
  }
  if ($age -gt $STALE_AFTER_S) {
    $form.Text = "STALE - this frame is $([int]$age)s old. The capture loop has stopped; do NOT aim against this."
  } elseif ($script:lastError) {
    $form.Text = "camera alignment - cannot read the preview: $($script:lastError)"
  } else {
    $form.Text = "camera alignment - aim until LIVE matches REFERENCE, then close this window"
  }
})
$timer.Start()

$form.Add_FormClosing({
  $timer.Stop()
  if ($box.Image) { $box.Image.Dispose() }
  New-Item -ItemType File -Path $stopFlag -Force | Out-Null
})

[void]$form.ShowDialog()
