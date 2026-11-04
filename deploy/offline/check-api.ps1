$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Get-Content -Raw -LiteralPath check_api.py | docker compose --env-file .env.offline -f compose.yaml -p issue-filter-offline exec -T issue-filter python -
if ($LASTEXITCODE -ne 0) { throw 'Offline API check failed; inspect docker compose logs' }
