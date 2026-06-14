$action = New-ScheduledTaskAction -Execute "C:\Users\Niravi\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe" -Argument "dashboard --host 0.0.0.0 --insecure" -WorkingDirectory "C:\Users\Niravi"
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "Niravi"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId "Niravi" -LogonType S4U -RunLevel Limited
Register-ScheduledTask -TaskName "Hermes Dashboard" -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force
Write-Output "✅ Task 'Hermes Dashboard' created!"
Start-ScheduledTask -TaskName "Hermes Dashboard" 2>$null
Write-Output "✅ Started now!"
Get-ScheduledTask -TaskName "Hermes Dashboard" | Format-List TaskName,State,Enabled
