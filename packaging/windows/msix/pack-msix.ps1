# Pack a MSIX layout directory into a .msix package using the Windows SDK.
#
# Usage: pack-msix.ps1 -LayoutDir <dir> -OutputFile <file.msix>
param(
  [Parameter(Mandatory = $true)][string]$LayoutDir,
  [Parameter(Mandatory = $true)][string]$OutputFile
)

$ErrorActionPreference = "Stop"

$exe = $null

# Prefer the newest x64 MakeAppx from the installed Windows SDK.
$candidates = Get-ChildItem -Path "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\makeappx.exe" `
  -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
if ($candidates) { $exe = $candidates[0].FullName }

if (-not $exe) {
  $cmd = Get-Command makeappx.exe -ErrorAction SilentlyContinue
  if ($cmd) { $exe = $cmd.Source }
}

if (-not $exe) {
  throw "makeappx.exe not found. Install the Windows SDK, or add MakeAppx to PATH."
}

Write-Host "Using MakeAppx: $exe"
& $exe pack /d "$LayoutDir" /p "$OutputFile" /o
if ($LASTEXITCODE -ne 0) {
  throw "MakeAppx failed with exit code $LASTEXITCODE"
}

Write-Host "Created $OutputFile"
