# infra/scripts/firewall.ps1
# Requires Administrator privileges to configure Windows Firewall inbound rules

# Check for Administrator rights
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "[-] Error: This script must be run as Administrator! Right-click PowerShell and select 'Run as administrator'." -ForegroundColor Red
    Exit
}

Write-Host "[+] Configuring Windows Firewall rules for Vision Assistant..." -ForegroundColor Cyan

# Define ports
$ports = @(
    @{ Name = "VA-Backend-API"; Port = "8000"; Protocol = "TCP" },
    @{ Name = "VA-MediaMTX-WHIP"; Port = "8889"; Protocol = "TCP" },
    @{ Name = "VA-WebRTC-UDP"; Port = "8189"; Protocol = "UDP" }
)

foreach ($rule in $ports) {
    # Remove existing rule if it exists to avoid duplicates
    Remove-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue

    # Create new inbound rule for the Private profile
    New-NetFirewallRule `
        -DisplayName $rule.Name `
        -Direction Inbound `
        -Profile Private `
        -Protocol $rule.Protocol `
        -LocalPort $rule.Port `
        -Action Allow | Out-Null

    Write-Host "    [OK] Allowed inbound $($rule.Protocol) port $($rule.Port) ($($rule.Name))" -ForegroundColor Green
}

Write-Host "[+] Firewall rules applied successfully!" -ForegroundColor Green