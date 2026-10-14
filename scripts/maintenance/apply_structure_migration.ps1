$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
function SafePath([string]$relative) {
    if ([IO.Path]::IsPathRooted($relative)) { throw 'Expected relative path' }
    $full = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $full" }
    return $full
}
$audit = SafePath 'docs/architecture/migration_20261012'
$plan = Get-Content -LiteralPath (Join-Path $audit 'plan.json') -Raw -Encoding utf8 | ConvertFrom-Json
$journal = Join-Path $audit 'moves.jsonl'
if (Test-Path -LiteralPath $journal) { throw 'Migration journal exists. Inspect before resuming; do not replay blindly.' }
$records = @()
# Preflight all targets before any movement.
foreach ($action in $plan.actions) {
    $src = SafePath $action.old
    $dst = SafePath $action.new
    if (-not (Test-Path -LiteralPath $src)) { throw "Missing: $src" }
    if (Test-Path -LiteralPath $dst) { throw "Destination exists: $dst" }
    $item = Get-Item -LiteralPath $src -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Unexpected link: $src" }
}
foreach ($action in $plan.actions) {
    $src = SafePath $action.old
    $dst = SafePath $action.new
    $item = Get-Item -LiteralPath $src -Force
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $src -Recurse -File -Force) } else { @($item) }
    $batch = @()
    foreach ($file in $files) {
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Nested link found' }
        $suffix = if ($item.PSIsContainer) { $file.FullName.Substring($src.Length).TrimStart('\') } else { '' }
        $newFile = if ($suffix) { Join-Path $dst $suffix } else { $dst }
        $batch += [pscustomobject]@{old=$file.FullName.Substring($root.Length+1);new=$newFile.Substring($root.Length+1);bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
    }
    # Persist hashes BEFORE the move for recovery after interruption.
    foreach ($record in $batch) { $record | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $audit 'files.jsonl') -Encoding utf8 }
    New-Item -ItemType Directory -Path (Split-Path $dst) -Force | Out-Null
    Move-Item -LiteralPath $src -Destination $dst
    if ($action.kind -eq 'junction') { New-Item -ItemType Junction -Path $src -Target $dst | Out-Null }
    elseif ($action.kind -eq 'hardlink') { New-Item -ItemType HardLink -Path $src -Target $dst | Out-Null }
    # Document redirects and relative links are finalized in a separate step.
    foreach ($record in $batch) {
        if ((Get-FileHash -LiteralPath (SafePath $record.new) -Algorithm SHA256).Hash -ne $record.sha256) { throw "Hash mismatch: $($record.new)" }
    }
    $action | ConvertTo-Json -Compress | Add-Content -LiteralPath $journal -Encoding utf8
    Write-Output "VERIFIED $($action.old) -> $($action.new) [$($files.Count) files]"
}
Write-Output 'Migration moves completed; finalize documentation next.'
