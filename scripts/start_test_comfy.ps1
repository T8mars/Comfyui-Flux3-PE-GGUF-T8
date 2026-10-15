param([int]$Port = 8267, [string]$ComfyUI = 'E:\comfyui-t8-onekey-5x\ComfyUI', [string]$Python = 'E:\comfyui-t8-onekey-5x\python\python.exe')
$ErrorActionPreference = 'Stop'
# Python's native argv and file APIs preserve spaces, brackets and Unicode.
# Start-Process expands wildcard characters even in redirected file paths.
& $Python (Join-Path $PSScriptRoot 'launch_comfy.py') --comfy-ui $ComfyUI --port $Port
if ($LASTEXITCODE -ne 0) { throw "Isolated ComfyUI launch failed: exit=$LASTEXITCODE" }
