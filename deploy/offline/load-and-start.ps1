param([switch]$LoadOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

Get-Command docker -ErrorAction Stop | Out-Null
foreach ($line in Get-Content -LiteralPath SHA256SUMS) {
    if (-not $line.Trim()) { continue }
    $parts = $line -split '  ', 2
    if ($parts.Count -ne 2) { throw 'Invalid checksum manifest' }
    if ($parts[1] -notmatch '^(images|host-tools)/') { continue }
    $actual = (Get-FileHash -LiteralPath $parts[1] -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $parts[0]) { throw "Checksum mismatch: $($parts[1])" }
}
docker load --input images/issue-filter-offline-20261104-cpu.tar
if ($LASTEXITCODE -ne 0) { throw 'docker load failed' }
if ($LoadOnly) { exit 0 }
docker compose version
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop with Compose is required on Windows' }
docker compose --env-file .env.offline -f compose.yaml -p issue-filter-offline up --pull never -d
if ($LASTEXITCODE -ne 0) { throw 'Service startup failed' }
Write-Host 'Started. Check readiness with: .\check-api.ps1'
