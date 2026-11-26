param([Parameter(Mandatory=$true)][string]$Root,[int]$EgoPort=9099,[int]$CommandPort=9093,[double]$TimeoutSeconds=0.6)
# Local host relay for a simulator bound to Windows loopback. No ROS dependency.
# Command steering is normalized LEFT-positive; invert exactly once for26R1.
$ErrorActionPreference='Stop'
[System.IO.Directory]::CreateDirectory($Root)|Out-Null
$utf8=New-Object System.Text.UTF8Encoding($false)
function Write-Atomic($Name,$Value) {
 $path=Join-Path $Root $Name;$tmp=$path+'.tmp'
 [System.IO.File]::WriteAllText($tmp,($Value|ConvertTo-Json -Depth 8 -Compress),$utf8)
 if([System.IO.File]::Exists($path)){[System.IO.File]::Replace($tmp,$path,[System.Management.Automation.Language.NullString]::Value)}else{[System.IO.File]::Move($tmp,$path)}
}
function Send-Command([double]$Throttle,[double]$Brake,[double]$Steer,[byte]$Gear) {
 if([double]::IsNaN($Throttle) -or [double]::IsNaN($Brake) -or [double]::IsNaN($Steer) -or $Throttle -lt 0 -or $Throttle -gt 1 -or $Brake -lt 0 -or $Brake -gt 1 -or [math]::Abs($Steer) -gt 1){throw 'Invalid finite command'}
 $stream=New-Object System.IO.MemoryStream;$writer=New-Object System.IO.BinaryWriter($stream)
 $writer.Write([System.Text.Encoding]::ASCII.GetBytes('#MoraiCtrlCmd$'));$writer.Write([int]27);$writer.Write((New-Object byte[] 12))
 $writer.Write([byte]2);$writer.Write($Gear);$writer.Write([byte]1)
 foreach($v in @(0,0,$Throttle,$Brake,(-$Steer),0)){$writer.Write([single]$v)}
 $writer.Write([byte]13);$writer.Write([byte]10);$writer.Flush();$packet=$stream.ToArray()
 if($packet.Length -ne 59){throw '26R1 command must be59bytes'}
 [void]$sender.Send($packet,$packet.Length,'127.0.0.1',$CommandPort);$writer.Dispose();$stream.Dispose()
}
$receiver=New-Object System.Net.Sockets.UdpClient($EgoPort)
$sender=New-Object System.Net.Sockets.UdpClient
$clock=[System.Diagnostics.Stopwatch]::StartNew();$lastSequence='';$lastChanged=-100.;$lastCommand=$null;$disarmUntil=-100.;$lastStateWrite=-100.;$packets=0;$commands=0;$watchdog=0
Write-Atomic 'relay_started.json' @{utc=[DateTime]::UtcNow.ToString('o');pid=$PID;ego_port=$EgoPort;command_port=$CommandPort;timeout_seconds=$TimeoutSeconds;steering='left-positive normalized; wire inverted once';packet_bytes=59}
try {
 while(-not [System.IO.File]::Exists((Join-Path $Root 'STOP'))) {
  $now=$clock.Elapsed.TotalSeconds
  if($receiver.Available -gt 0){
   do {$peer=New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any,0);$data=$receiver.Receive([ref]$peer);$packets++}while($receiver.Available -gt 0)
   if($data.Length -ne 245 -or [System.Text.Encoding]::ASCII.GetString($data,0,11) -ne '#MoraiInfo$' -or [BitConverter]::ToInt32($data,11) -ne 216){throw 'Unexpected26R1 Ego packet schema'}
   if($now-$lastStateWrite -ge .035){
    $b=27;$state=@{received_unix_ms=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds();packet_length=$data.Length;native_sec=[BitConverter]::ToUInt32($data,$b);native_nsec=[BitConverter]::ToUInt32($data,$b+4);ctrl_mode=$data[$b+8];gear=$data[$b+9];signed_speed_kph=[BitConverter]::ToSingle($data,$b+10);position=@{x=[BitConverter]::ToSingle($data,$b+50);y=[BitConverter]::ToSingle($data,$b+54);z=[BitConverter]::ToSingle($data,$b+58)};rotation=@{x=[BitConverter]::ToSingle($data,$b+62);y=[BitConverter]::ToSingle($data,$b+66);z=[BitConverter]::ToSingle($data,$b+70)};velocity_kph=@{x=[BitConverter]::ToSingle($data,$b+74);y=[BitConverter]::ToSingle($data,$b+78);z=[BitConverter]::ToSingle($data,$b+82)};front_steer_deg=[BitConverter]::ToSingle($data,$b+110);rear_steer_deg=[BitConverter]::ToSingle($data,$b+114);link_id=[System.Text.Encoding]::UTF8.GetString($data,$b+118,38).Trim([char]0);packet_count=$packets}
    Write-Atomic 'state.json' $state;$lastStateWrite=$now
   }
  }
  $path=Join-Path $Root 'command.json'
  if([System.IO.File]::Exists($path)){
   try {$inputCommand=[System.IO.File]::ReadAllText($path)|ConvertFrom-Json} catch {$inputCommand=$null}
   if($null -ne $inputCommand -and [string]$inputCommand.sequence -ne $lastSequence){
    $lastSequence=[string]$inputCommand.sequence;$lastChanged=$now;$lastCommand=$inputCommand
    if(-not $lastCommand.armed){$disarmUntil=$now+.1}
   }
  }
  $active=($null -ne $lastCommand -and $lastCommand.armed)
  if($active -and $now-$lastChanged -le $TimeoutSeconds){Send-Command $lastCommand.throttle $lastCommand.brake $lastCommand.steer ([byte]$lastCommand.gear);$commands++}
  elseif($active -or $now -lt $disarmUntil){Send-Command 0 1 0 1;$watchdog++}
  if(($packets+$commands)%40 -eq 0){Write-Atomic 'relay_status.json' @{utc=[DateTime]::UtcNow.ToString('o');packets=$packets;commands=$commands;watchdog_packets=$watchdog;command_armed=$active;command_age_s=$now-$lastChanged}}
  Start-Sleep -Milliseconds 15
 }
} finally {
 for($i=0;$i -lt 15;$i++){try{Send-Command 0 1 0 1}catch{};Start-Sleep -Milliseconds 20}
 Write-Atomic 'relay_stopped.json' @{utc=[DateTime]::UtcNow.ToString('o');packets=$packets;commands=$commands;watchdog_packets=$watchdog}
 $receiver.Close();$sender.Close()
}
