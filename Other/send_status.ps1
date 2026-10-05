# Send Other/status_email.txt to yourself over SMTP. Run it yourself; credentials never leave this machine.
#   powershell -ExecutionPolicy Bypass -File Other\send_status.ps1 -SmtpServer smtp.example.com
# Other/email.txt: either two lines (address, then password) or KEY=VALUE lines
# with keys like user/email/username and pass/password.
param(
    [Parameter(Mandatory = $true)][string]$SmtpServer,
    [int]$Port = 587,
    [string]$CredFile = (Join-Path $PSScriptRoot "email.txt"),
    [string]$BodyFile = (Join-Path $PSScriptRoot "status_email.txt")
)
$ErrorActionPreference = "Stop"
$lines = Get-Content $CredFile | Where-Object { $_.Trim() }
$kv = @{}
foreach ($l in $lines) { if ($l -match '^\s*([A-Za-z_]+)\s*[=:]\s*(.+?)\s*$') { $kv[$matches[1].ToLower()] = $matches[2] } }
$user = @($kv['user'], $kv['email'], $kv['username'], $kv['address']) | Where-Object { $_ } | Select-Object -First 1
$pass = @($kv['pass'], $kv['password'], $kv['pwd']) | Where-Object { $_ } | Select-Object -First 1
if (-not $user -or -not $pass) { $user = $lines[0].Trim(); $pass = $lines[1].Trim() }

$text = Get-Content $BodyFile -Raw -Encoding UTF8
$subject = "road_htr status"
if ($text -match '^Subject:\s*(.+?)\r?\n') { $subject = $matches[1]; $text = $text.Substring($matches[0].Length) }

$cred = New-Object System.Management.Automation.PSCredential($user, (ConvertTo-SecureString $pass -AsPlainText -Force))
Send-MailMessage -SmtpServer $SmtpServer -Port $Port -UseSsl -Credential $cred `
    -From $user -To $user -Subject $subject -Body $text -Encoding UTF8
Write-Output "Sent '$subject' to $user"
