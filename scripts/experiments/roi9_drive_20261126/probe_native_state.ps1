param([Parameter(Mandatory=$true)][string]$Receipt,[int]$EgoPort=9099)
# Read one26R1 Ego packet after the owned relay stops. Sends no control packet.
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $Receipt){throw 'Preserve the previous native observation'}
$receiver=New-Object System.Net.Sockets.UdpClient($EgoPort)
$receiver.Client.ReceiveTimeout=2500
try {
 $peer=New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any,0)
 $data=$receiver.Receive([ref]$peer)
 if($data.Length -ne245 -or [System.Text.Encoding]::ASCII.GetString($data,0,11) -ne '#MoraiInfo$'){throw 'Unexpected26R1 Ego packet'}
 $b=27;$p=@(Get-Process -Name Simulator)
 Add-Type @'
using System;using System.Runtime.InteropServices;
public static class NativeWindowProbe { [DllImport("user32.dll")]public static extern bool IsIconic(IntPtr h);[DllImport("user32.dll")]public static extern bool IsZoomed(IntPtr h); }
'@
 if($p.Count -ne1){throw 'Expected one Simulator process'}
 $value=@{utc=[DateTime]::UtcNow.ToString('o');packet_length=$data.Length;ctrl_mode=$data[$b+8];gear=$data[$b+9];signed_speed_kph=[BitConverter]::ToSingle($data,$b+10);position=@{x=[BitConverter]::ToSingle($data,$b+50);y=[BitConverter]::ToSingle($data,$b+54);z=[BitConverter]::ToSingle($data,$b+58)};simulator_pid=$p[0].Id;responding=$p[0].Responding;window_minimized=[NativeWindowProbe]::IsIconic($p[0].MainWindowHandle);window_maximized=[NativeWindowProbe]::IsZoomed($p[0].MainWindowHandle);control_commands_sent=0;scope='Read-only UDP/window observation after owned relay shutdown'}
 $value|ConvertTo-Json -Depth 6|Set-Content -Encoding UTF8 $Receipt
} finally {$receiver.Close()}
