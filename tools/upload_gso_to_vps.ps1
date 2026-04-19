param(
    [Parameter(Mandatory = $true)]
    [string]$SshTarget,

    [string]$LocalZip = "artifacts/gso_preprocessed.zip",
    [string]$RemoteDir = "~/datasets/gso_preprocessed",
    [int]$Port = 22,
    [string]$IdentityFile = "",
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    throw "ssh command not found. Install OpenSSH client first."
}
if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    throw "scp command not found. Install OpenSSH client first."
}

$resolvedZip = Resolve-Path -Path $LocalZip -ErrorAction Stop
$zipLeaf = [System.IO.Path]::GetFileName($resolvedZip.Path)
$remoteZip = "$RemoteDir/$zipLeaf"

$sshBaseArgs = @("-p", "$Port")
if ($IdentityFile -ne "") {
    $sshBaseArgs += @("-i", $IdentityFile)
}

$scpBaseArgs = @("-P", "$Port")
if ($IdentityFile -ne "") {
    $scpBaseArgs += @("-i", $IdentityFile)
}

Write-Host "Ensuring remote directory exists: $RemoteDir"
& ssh @sshBaseArgs $SshTarget "mkdir -p '$RemoteDir'"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create remote directory."
}

Write-Host "Uploading zip to $SshTarget:$remoteZip"
& scp @scpBaseArgs $resolvedZip.Path "$SshTarget:$remoteZip"
if ($LASTEXITCODE -ne 0) {
    throw "Upload failed."
}

if (-not $SkipExtract) {
    $extractDir = "$RemoteDir/data"
    Write-Host "Extracting zip on remote host to: $extractDir"
    & ssh @sshBaseArgs $SshTarget "mkdir -p '$extractDir' && unzip -oq '$remoteZip' -d '$extractDir'"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote extraction failed."
    }

    Write-Host "Remote extraction completed."
    & ssh @sshBaseArgs $SshTarget "echo 'Remote directory overview:' && ls -lah '$RemoteDir' && echo 'Extracted dataset overview:' && ls -lah '$extractDir'"
    if ($LASTEXITCODE -ne 0) {
        throw "Remote verification failed."
    }
}

Write-Host "Transfer workflow complete."
Write-Host "Zip path on VPS: $remoteZip"
if (-not $SkipExtract) {
    Write-Host "Extracted root on VPS: $RemoteDir/data"
}
