param(
    [int]$StartLayout = 0,
    [int]$Count = 1,
    [string]$LayoutFile = "",
    [ValidateSet("fast", "realtime")]
    [string]$Mode = "fast",
    [switch]$Visible,
    [switch]$LeaveOpen,
    [switch]$OfficialThread,
    [switch]$DebugController,
    [switch]$SaveImages,
    [switch]$ListOnly,
    [string]$RunId = "",
    [int]$TimeoutSeconds = 260,
    [string]$WebotsPath = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$debugDir = Join-Path $repoRoot "controllers\main\assignment\debug"
$scoreLog = Join-Path $debugDir "sim_progress.log"
$assignmentLog = Join-Path $debugDir "assignment_debug.log"
$logsRoot = Join-Path $PSScriptRoot "generated_layout_logs"

if ($RunId -eq "") {
    $RunId = "$(Get-Date -Format 'yyyyMMdd_HHmmss')_pid$PID"
}

$logsDir = Join-Path $logsRoot $RunId
$resultsPath = Join-Path $logsDir "generated_layout_results.csv"
$latestResultsPath = Join-Path $PSScriptRoot "generated_layout_results_latest.csv"
$manifestPath = Join-Path $logsDir "run_manifest.txt"

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

if (-not (Test-Path -LiteralPath $WebotsPath)) {
    throw "Webots executable not found: $WebotsPath"
}

function Get-LayoutIds {
    if ($LayoutFile -ne "") {
        $layoutPath = if ([System.IO.Path]::IsPathRooted($LayoutFile)) {
            $LayoutFile
        } else {
            Join-Path $repoRoot $LayoutFile
        }

        $ids = Get-Content -LiteralPath $layoutPath |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -ne "" -and -not $_.StartsWith("#") } |
            ForEach-Object { [int]$_ }

        if ($Count -gt 0) {
            return @($ids | Select-Object -First $Count)
        }

        return @($ids)
    }

    if ($Count -le 0) {
        $Count = 1
    }

    $ids = @()
    for ($i = 0; $i -lt $Count; $i++) {
        $ids += ($StartLayout + $i)
    }
    return $ids
}

function Get-LastMatchingLine {
    param(
        [string[]]$Lines,
        [string]$Pattern
    )

    $match = ""
    foreach ($line in $Lines) {
        if ($line -match $Pattern) {
            $match = $line
        }
    }
    return $match
}

function Parse-SimulatorLog {
    param(
        [string]$Path,
        [int]$LayoutId,
        [int]$ExitCode,
        [bool]$TimedOut,
        [bool]$LeftOpen,
        [string]$StdoutLog,
        [string]$StderrLog,
        [string]$ScoreLogCopy
        ,
        [string]$AssignmentLogCopy
    )

    $matrix = @()
    for ($lap = 0; $lap -lt 3; $lap++) {
        $row = @()
        for ($gate = 0; $gate -lt 5; $gate++) {
            $row += $false
        }
        $matrix += ,$row
    }

    $layout = @()
    $lapTimes = "missing"
    $finished = $false
    $debugTimeout = $false
    $lastLine = ""
    $finalProgressLine = ""

    if (Test-Path -LiteralPath $Path) {
        $lines = Get-Content -LiteralPath $Path
        if ($lines.Count -gt 0) {
            $lastLine = $lines[-1]
        }

        foreach ($line in $lines) {
            if ($line -match "^gate_truth\s+(\d+)\s+pos=\(([^)]+)\)\s+size=\(([^)]+)\)\s+yaw=([-0-9.]+)") {
                $layout += "G$($Matches[1]) pos=($($Matches[2])) size=($($Matches[3])) yaw=$($Matches[4])"
            }

            if ($line -match "gate_reached gate=(\d+).*lap=(\d+)") {
                $gate = [int]$Matches[1]
                $lap = [int]$Matches[2]
                if ($lap -ge 0 -and $lap -lt 3 -and $gate -ge 0 -and $gate -lt 5) {
                    $matrix[$lap][$gate] = $true
                }
            }

            if ($line -match "\bfinished\b" -or $line -match "\bdebug_finished\b") {
                $finished = $true
            }

            if ($line -match "\bdebug_timeout\b") {
                $debugTimeout = $true
            }

            if ($line -match "lap_times=\[([^\]]+)\]") {
                $lapTimes = $Matches[1]
            }
        }

        $finalProgressLine = Get-LastMatchingLine -Lines $lines -Pattern "gate_progress="
    }

    $lapCounts = @()
    $total = 0
    for ($lap = 0; $lap -lt 3; $lap++) {
        $count = 0
        for ($gate = 0; $gate -lt 5; $gate++) {
            if ($matrix[$lap][$gate]) {
                $count += 1
            }
        }
        $lapCounts += $count
        $total += $count
    }

    $passed = ($total -eq 15 -and $finished)
    $reason = "failed_incomplete_gate_progress"
    if ($passed) {
        $reason = "passed_all_15_webots_gate_checks"
    } elseif (-not (Test-Path -LiteralPath $Path)) {
        $reason = "failed_missing_simulator_log"
    } elseif ($debugTimeout -or $TimedOut) {
        $reason = "failed_timeout_before_all_gates"
    } elseif ($finished -and $total -lt 15) {
        $reason = "failed_finished_but_missed_gate"
    }

    [PSCustomObject]@{
        LayoutId = $LayoutId
        Passed = $passed
        Reason = $reason
        TotalGates = $total
        Lap0Gates = $lapCounts[0]
        Lap1Gates = $lapCounts[1]
        Lap2Gates = $lapCounts[2]
        Finished = $finished
        TimedOut = $TimedOut
        LeftOpen = $LeftOpen
        ExitCode = $ExitCode
        Mode = $(if ($Visible) { "visible-$Mode" } else { "hidden-$Mode" })
        OfficialThread = [bool]$OfficialThread
        LapTimes = $lapTimes
        Layout = ($layout -join " | ")
        LastScoreLine = $lastLine
        FinalProgressLine = $finalProgressLine
        ScoreLog = $ScoreLogCopy
        AssignmentLog = $AssignmentLogCopy
        StdoutLog = $StdoutLog
        StderrLog = $StderrLog
    }
}

function Stop-StartedWebotsProcess {
    param(
        [System.Diagnostics.Process]$Process
    )

    if ($null -eq $Process) {
        return
    }

    $childIds = @()
    try {
        $childIds = Get-CimInstance Win32_Process |
            Where-Object { $_.ParentProcessId -eq $Process.Id } |
            Select-Object -ExpandProperty ProcessId
    } catch {
        $childIds = @()
    }

    foreach ($childId in $childIds) {
        Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue
    }

    if (-not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

function Copy-IfPresent {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (Test-Path -LiteralPath $Source) {
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
        return $true
    }
    return $false
}

$layoutIds = Get-LayoutIds

if ($ListOnly) {
    Write-Host "Selected layout IDs: $($layoutIds -join ', ')"
    exit 0
}

$manifest = @(
    "run_id=$RunId",
    "started_at=$(Get-Date -Format o)",
    "repo_root=$repoRoot",
    "mode=$Mode",
    "visible=$([bool]$Visible)",
    "leave_open=$([bool]$LeaveOpen)",
    "official_thread=$([bool]$OfficialThread)",
    "debug_controller=$([bool]$DebugController)",
    "save_images=$([bool]$SaveImages)",
    "timeout_seconds=$TimeoutSeconds",
    "layout_ids=$($layoutIds -join ',')",
    "webots_path=$WebotsPath"
)
$manifest | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$results = @()

foreach ($layoutId in $layoutIds) {
    $label = "{0:D4}" -f $layoutId
    $layoutDir = Join-Path $logsDir "layout_$label"
    New-Item -ItemType Directory -Force -Path $layoutDir | Out-Null

    $stdoutLog = Join-Path $layoutDir "stdout.log"
    $stderrLog = Join-Path $layoutDir "stderr.log"
    $scoreCopy = Join-Path $layoutDir "sim_progress.log"
    $assignmentCopy = Join-Path $layoutDir "assignment_debug.log"
    $layoutManifest = Join-Path $layoutDir "layout_manifest.txt"

    Remove-Item -LiteralPath $stdoutLog, $stderrLog, $scoreCopy, $assignmentCopy -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $scoreLog, $assignmentLog -ErrorAction SilentlyContinue

    $layoutRunId = "${RunId}_layout_${label}"
    $env:MICRO502_RANDOM_SEED = [string]$layoutId
    $env:MICRO502_RUN_ID = $layoutRunId
    $env:MICRO502_MAX_TIME = "245"
    $env:MICRO502_DEBUG = $(if ($DebugController) { "1" } else { "0" })
    $env:MICRO502_SAVE_IMAGES = $(if ($SaveImages) { "1" } else { "0" })
    $env:MICRO502_DEBUG_STDOUT = "0"
    $env:MICRO502_SYNC_ASSIGNMENT = $(if ($OfficialThread) { "0" } else { "1" })

    @(
        "run_id=$layoutRunId",
        "layout_id=$layoutId",
        "started_at=$(Get-Date -Format o)",
        "score_log_live=$scoreLog",
        "assignment_log_live=$assignmentLog",
        "score_log_archive=$scoreCopy",
        "assignment_log_archive=$assignmentCopy"
    ) | Set-Content -LiteralPath $layoutManifest -Encoding UTF8

    $arguments = if ($Visible) {
        @("--mode=$Mode", "--stdout", "--stderr", "worlds\crazyflie_world_assignment.wbt")
    } else {
        @("--batch", "--mode=$Mode", "--no-rendering", "--stdout", "--stderr", "worlds\crazyflie_world_assignment.wbt")
    }

    Write-Host "Running layout $layoutId mode=$Mode visible=$([bool]$Visible) officialThread=$([bool]$OfficialThread)"
    $startInfo = @{
        FilePath = $WebotsPath
        ArgumentList = $arguments
        WorkingDirectory = $repoRoot
        RedirectStandardOutput = $stdoutLog
        RedirectStandardError = $stderrLog
        PassThru = $true
    }
    if (-not $Visible) {
        $startInfo.WindowStyle = "Hidden"
    }

    $proc = Start-Process @startInfo
    $timedOut = $false
    $scoreFinished = $false
    $leftOpenNow = $false
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            break
        }

        if (Test-Path -LiteralPath $scoreLog) {
            $tail = Get-Content -LiteralPath $scoreLog -Tail 8 -ErrorAction SilentlyContinue
            if ($tail -match "\bfinished\b" -or $tail -match "\bdebug_finished\b" -or $tail -match "\bdebug_timeout\b") {
                $scoreFinished = $true
                break
            }
        }

        Start-Sleep -Seconds 1
    }

    if ($scoreFinished -and -not $proc.HasExited) {
        Start-Sleep -Seconds 2
        if ($Visible -and $LeaveOpen) {
            $leftOpenNow = $true
        } else {
            Stop-StartedWebotsProcess -Process $proc
        }
    }

    if (-not $scoreFinished -and -not $proc.HasExited) {
        $timedOut = $true
        if ($Visible -and $LeaveOpen) {
            $leftOpenNow = $true
        } else {
            Stop-StartedWebotsProcess -Process $proc
        }
    }

    $scoreCopied = Copy-IfPresent -Source $scoreLog -Destination $scoreCopy
    $assignmentCopied = Copy-IfPresent -Source $assignmentLog -Destination $assignmentCopy

    @(
        "ended_at=$(Get-Date -Format o)",
        "timed_out=$timedOut",
        "score_finished=$scoreFinished",
        "left_open=$leftOpenNow",
        "score_copied=$scoreCopied",
        "assignment_copied=$assignmentCopied",
        "archive_is_final=$(-not $leftOpenNow)"
    ) | Add-Content -LiteralPath $layoutManifest -Encoding UTF8

    $exitCode = if ($leftOpenNow) {
        0
    } elseif ($timedOut) {
        -1
    } elseif ($scoreFinished) {
        0
    } else {
        $proc.ExitCode
    }

    $result = Parse-SimulatorLog `
        -Path $scoreCopy `
        -LayoutId $layoutId `
        -ExitCode $exitCode `
        -TimedOut $timedOut `
        -LeftOpen $leftOpenNow `
        -StdoutLog $stdoutLog `
        -StderrLog $stderrLog `
        -ScoreLogCopy $scoreCopy `
        -AssignmentLogCopy $assignmentCopy

    if ($leftOpenNow) {
        $result.Passed = $false
        $result.Reason = "not_final_left_open_snapshot"
    }

    $results += $result
    $result | Format-List LayoutId, Passed, Reason, TotalGates, Lap0Gates, Lap1Gates, Lap2Gates, Finished, TimedOut, LeftOpen, LapTimes, ScoreLog, AssignmentLog
}

$results | Export-Csv -NoTypeInformation -Path $resultsPath
Copy-Item -LiteralPath $resultsPath -Destination $latestResultsPath -Force

@(
    "ended_at=$(Get-Date -Format o)",
    "results_path=$resultsPath",
    "latest_results_path=$latestResultsPath"
) | Add-Content -LiteralPath $manifestPath -Encoding UTF8

$passedCount = @($results | Where-Object { $_.Passed }).Count
$failedCount = $results.Count - $passedCount
Write-Host "Generated-layout tests complete: passed=$passedCount failed=$failedCount results=$resultsPath"

if ($failedCount -gt 0) {
    exit 1
}
