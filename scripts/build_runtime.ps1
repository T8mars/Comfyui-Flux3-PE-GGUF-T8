$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $projectRoot 'vendor\llama.cpp'
$buildPath = Join-Path $sourcePath 'build-pe'
$runtimePath = Join-Path $projectRoot 'runtime'
$binPath = Join-Path $runtimePath 'bin'
$upstreamPath = Join-Path $runtimePath 'upstream'
$commit = '620d2225cc436ab0b001e3200e590e3b286ef9ab'
$assets = @(
    @{Name='llama-b11669-bin-win-cuda-13.4-x64.zip'; Hash='3b92d5c064f5032e603e531d3ec8961f3bd6613e5dfad047399a92dec8e95dbc'; Folder=$upstreamPath},
    @{Name='cudart-llama-bin-win-cuda-13.4-x64.zip'; Hash='738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668'; Folder=$binPath}
)
if (-not (Test-Path -LiteralPath $sourcePath)) {
    git clone --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp.git $sourcePath
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp clone failed' }
}
if ((git -C $sourcePath rev-parse HEAD) -ne $commit) { throw 'Unexpected llama.cpp revision' }
$patchPath = Join-Path $projectRoot 'patches\llama-generated-penalties.patch'
git -C $sourcePath apply --reverse --check $patchPath 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $sourcePath apply --check $patchPath
    if ($LASTEXITCODE -ne 0) { throw 'Runtime patch does not apply cleanly' }
    git -C $sourcePath apply $patchPath
    if ($LASTEXITCODE -ne 0) { throw 'Runtime patch failed' }
}
New-Item -ItemType Directory -Force -Path $runtimePath,$binPath,$upstreamPath | Out-Null
foreach ($asset in $assets) {
    $zipPath = Join-Path $runtimePath $asset.Name
    if (-not (Test-Path -LiteralPath $zipPath)) {
        Invoke-WebRequest -Uri ('https://github.com/ggml-org/llama.cpp/releases/download/b11669/'+$asset.Name) -OutFile $zipPath
    }
    if ((Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $asset.Hash) {
        throw ('Asset checksum failed: '+$asset.Name)
    }
    Expand-Archive -LiteralPath $zipPath -DestinationPath $asset.Folder -Force
}
cmake -S $sourcePath -B $buildPath -G 'Visual Studio 17 2022' -A x64 -DGGML_CUDA=OFF -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_MTMD=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF "-DFLUX3_PE_TEST_SOURCE=$projectRoot/tests/native/test-pe-sampling.cpp"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
cmake --build $buildPath --config Release --target llama-server llama-quantize test-pe-sampling --parallel 8
if ($LASTEXITCODE -ne 0) { throw 'Runtime build failed' }
Copy-Item -Path (Join-Path $buildPath 'bin\Release\*.dll'),(Join-Path $buildPath 'bin\Release\*.exe') -Destination $binPath
Copy-Item -LiteralPath (Join-Path $upstreamPath 'ggml-cuda.dll') -Destination (Join-Path $binPath 'ggml-cuda.dll')
$serverHash = (Get-FileHash -LiteralPath (Join-Path $binPath 'llama-server.exe') -Algorithm SHA256).Hash.ToLowerInvariant()
$coreHashes = @{}
foreach ($name in @('llama-common.dll','llama-server-impl.dll','llama.dll','mtmd.dll','ggml.dll','ggml-base.dll','ggml-cpu.dll','ggml-cuda.dll')) {
    $coreHashes[$name] = (Get-FileHash -LiteralPath (Join-Path $binPath $name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
@{source_commit=$commit;release='b11669';penalties_generated_only=$true;server_sha256=$serverHash;core_sha256=$coreHashes;cuda_backend='official CUDA 13.4 binary, same source commit'} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $binPath 'flux3-pe-runtime.json') -Encoding utf8NoBOM
& (Join-Path $binPath 'llama-server.exe') --list-devices
if ($LASTEXITCODE -ne 0) { throw 'Runtime device check failed' }
