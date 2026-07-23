[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Distro = "Ubuntu",
    [string]$RepoPath = "/home/gsx1339/aquant",
    [string]$CloseTime = "19:30",
    [string]$MorningTime = "08:30",
    [string]$TaskPrefix = "AQuant",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$closeTask = "$TaskPrefix-Daily-Close"
$morningTask = "$TaskPrefix-Daily-Morning-Recheck"
$taskNames = @($closeTask, $morningTask)

if ($Remove) {
    foreach ($taskName in $taskNames) {
        if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
            if ($PSCmdlet.ShouldProcess($taskName, "Unregister scheduled task")) {
                Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
            }
        }
    }
    return
}

if ($RepoPath.Contains("'")) {
    throw "RepoPath must not contain a single quote."
}

function Convert-ToTriggerTime {
    param([string]$Value)
    try {
        return [datetime]::ParseExact(
            $Value,
            "HH:mm",
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
    catch {
        throw "Time '$Value' must use 24-hour HH:mm format."
    }
}

function New-AQuantTask {
    param(
        [string]$TaskName,
        [string]$Mode,
        [string]$At
    )

    $logPath = "artifacts/daily-update/logs/scheduler-$Mode.log"
    $command = "cd '$RepoPath' && mkdir -p artifacts/daily-update/logs && " +
        "uv run --extra data python -u scripts/tushare_daily_update.py --mode $Mode " +
        "--workers 4 --interval 0.25 >> '$logPath' 2>&1"
    $arguments = "-d `"$Distro`" bash -lc `"$command`""
    $action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wsl.exe" `
        -Argument $arguments
    $trigger = New-ScheduledTaskTrigger -Daily -At (Convert-ToTriggerTime $At)
    $settings = New-ScheduledTaskSettingsSet `
        -WakeToRun `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 15) `
        -ExecutionTimeLimit (New-TimeSpan -Hours 3)
    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType Interactive `
        -RunLevel Limited
    $task = New-ScheduledTask `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "AQuant Tushare $Mode incremental data update"

    if ($PSCmdlet.ShouldProcess($TaskName, "Register scheduled task at $At")) {
        Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
    }
}

New-AQuantTask -TaskName $closeTask -Mode "close" -At $CloseTime
New-AQuantTask -TaskName $morningTask -Mode "morning" -At $MorningTime

Get-ScheduledTask -TaskName "$TaskPrefix-Daily-*" |
    Select-Object TaskName, State
