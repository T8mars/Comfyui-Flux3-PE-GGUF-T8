param([Parameter(Mandatory=$true)][string]$ComfyUI, [string]$UserDirectory, [string]$InputDirectory)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$comfyRoot = (Resolve-Path -LiteralPath $ComfyUI).Path
if (-not (Test-Path -LiteralPath (Join-Path $comfyRoot 'main.py') -PathType Leaf)) {
    throw 'ComfyUI must point to the directory containing main.py'
}
$nodePath = Join-Path $comfyRoot 'custom_nodes\ComfyUI-Flux3-PE'
# Registry and Git installations can use a different folder name.
if ((Split-Path -Parent $projectRoot) -eq (Join-Path $comfyRoot 'custom_nodes')) {
    $nodePath = $projectRoot
}
if (Test-Path -LiteralPath $nodePath) {
    $node = Get-Item -LiteralPath $nodePath
    if (($node.LinkType -eq 'Junction' -and $node.Target -eq $projectRoot) -or $node.FullName -eq $projectRoot) {
        Write-Output "Already installed: $nodePath"
    } else {
        throw "Destination already exists: $nodePath"
    }
} else {
    New-Item -ItemType Junction -Path $nodePath -Value $projectRoot | Out-Null
    Write-Output "Installed: $nodePath"
}
if (-not $UserDirectory) { $UserDirectory = Join-Path $comfyRoot 'user' }
if (-not $InputDirectory) { $InputDirectory = Join-Path $comfyRoot 'input' }
$workflowDirectory = Join-Path $UserDirectory 'default\workflows\FLUX3_PE'
$imageDirectory = $InputDirectory
New-Item -ItemType Directory -Path $workflowDirectory,$imageDirectory -Force | Out-Null
function Copy-Example($Source, $Destination) {
    if (Test-Path -LiteralPath $Destination) {
        if ((Get-FileHash -LiteralPath $Source).Hash -eq (Get-FileHash -LiteralPath $Destination).Hash) { return }
        $backupPath = "$Destination.$(Get-Date -Format 'yyyyMMdd-HHmmss-ffff').bak"
        Copy-Item -LiteralPath $Destination -Destination $backupPath
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'examples') -Filter '*.json' -File | Where-Object { $_.Name -notlike '*.api.json' } | ForEach-Object {
    Copy-Example $_.FullName (Join-Path $workflowDirectory $_.Name)
}
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'examples\reference_images') -Filter '*.png' -File | ForEach-Object {
    Copy-Example $_.FullName (Join-Path $imageDirectory ('FLUX3_PE_' + $_.Name))
}
Write-Output "Saved canvas workflows: $workflowDirectory"
Write-Output "Installed reference examples: $imageDirectory"
Write-Output 'Restart ComfyUI, then search for FLUX 3 Prompt Enhancement.'
