# Daily GitHub backup of the project (AGENTS.md "Backup").
#   powershell -ExecutionPolicy Bypass -File Other\scripts\daily_backup.ps1
# Pushes through the repository's deploy key (~/.ssh/road_htr_deploy, write access,
# scoped to this one repository). Git metadata lives in Other/backup.git so the project
# root keeps only AGENTS.md. Exclusions (secrets, checkpoints, venvs, competition data)
# are in Other/backup.git/info/exclude and Other/.gitignore.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$git = "C:\Program Files\Git\cmd\git.exe"
$gd = "Other\backup.git"
$env:GIT_TERMINAL_PROMPT = "0"
$log = Join-Path $root "Other\backup_log.txt"
function Say($text) { $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $text"; Write-Output $line; Add-Content -Path $log -Value $line }

# "When the CPU load is low": skip this run if the machine is busy.
$load = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
if ($load -gt 60) { Say "skipped: CPU load $load% is above 60%"; exit 0 }

& $git --git-dir=$gd --work-tree=. add -A
# Hard stop if a secret would ever be committed.
$staged = & $git --git-dir=$gd --work-tree=. diff --cached --name-only
$leak = @($staged | Where-Object { $_ -match '(^|/)\.env$|email\.txt$|\.pem$|road_htr_deploy' })
if ($leak.Count -gt 0) {
    & $git --git-dir=$gd --work-tree=. reset --quiet
    Say "ABORTED: secret file staged: $($leak -join ', ')"
    exit 1
}
if (@($staged).Count -eq 0) { Say "nothing changed since the last backup" }
else {
    & $git --git-dir=$gd --work-tree=. commit --quiet -m "Daily backup $(Get-Date -Format 'yyyy-MM-dd'): $(@($staged).Count) files changed"
    Say "committed $(@($staged).Count) changed files"
}
& $git --git-dir=$gd --work-tree=. push --quiet origin main
if ($LASTEXITCODE -ne 0) { Say "PUSH FAILED (exit $LASTEXITCODE)"; exit 1 }
Say "pushed: GitHub main at $(& $git --git-dir=$gd --work-tree=. rev-parse --short HEAD)"
