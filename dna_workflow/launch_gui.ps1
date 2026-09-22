[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$root = Split-Path -Parent $PSCommandPath
$python = 'D:\data\DNA DoS\workflow_runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Workflow runtime not found: $python" }
if (-not (Test-Path -LiteralPath (Join-Path $root 'gui_server.py'))) { throw 'gui_server.py not found.' }
$launchArgs = @('-u', (Join-Path $root 'gui_server.py'), '--host', '127.0.0.1', '--port', "$Port")
if (-not $NoBrowser) { $launchArgs += '--open-browser' }
& $python @launchArgs
