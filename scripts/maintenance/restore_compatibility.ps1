# Recreate missing local Windows compatibility paths after restoring canonical files.
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
function SafePath([string]$relative) {
    if ([IO.Path]::IsPathRooted($relative)) { throw 'Expected relative path' }
    $full = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $full.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Outside workspace' }
    return $full
}
$plan = Get-Content -LiteralPath (SafePath 'docs/architecture/migration_20261012/plan.json') -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($action in $plan.actions) {
    if ($action.kind -eq 'document') { continue }
    $old = SafePath $action.old
    $new = SafePath $action.new
    if (-not (Test-Path -LiteralPath $new)) { throw "Restore canonical file first: $new" }
    if (Test-Path -LiteralPath $old) { continue }
    New-Item -ItemType Directory -Path (Split-Path $old) -Force | Out-Null
    $type = if ($action.kind -eq 'junction') { 'Junction' } else { 'HardLink' }
    New-Item -ItemType $type -Path $old -Target $new | Out-Null
    Write-Output "Restored $($action.old)"
}
