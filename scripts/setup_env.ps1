# Sets JAVA_HOME for PySpark (edit path if your JDK is elsewhere)
$candidates = @(
    "D:\gym app\jdk\jdk-17.0.2",
    (Get-ChildItem "C:\Program Files\Microsoft\jdk-*" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName,
    (Get-ChildItem "C:\Program Files\Eclipse Adoptium\jdk-*" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
) | Where-Object { $_ -and (Test-Path "$_\bin\java.exe") }

$jdk = $candidates | Select-Object -First 1
if ($jdk) {
    $env:JAVA_HOME = $jdk
    $env:Path = "$env:JAVA_HOME\bin;" + $env:Path
    Write-Host "JAVA_HOME=$env:JAVA_HOME"
    java -version
} else {
    Write-Host "JDK 17 not found. Install Java 17+ or set JAVA_HOME manually."
}
