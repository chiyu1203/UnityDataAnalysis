# Usage in PowerShell as admin: .\move_files_to_subdirs.ps1 -StartID 25001 -ExperimentDate 250624
param (
    [Parameter(Mandatory=$true)][int]$StartID,
    [Parameter(Mandatory=$true)][string]$ExperimentDate
)

# Hardcoded parameters
$ExperimentName = "choices"
$SessionNumber = "session1"
# Base directory
$BaseDir = "C:\Users\neuroPC\Documents\test"

# Get subdirectories under BaseDir, excluding those starting with "GN"
$Subdirs = Get-ChildItem -Path $BaseDir | Where-Object { $_.PSIsContainer -and ($_.Name -notlike "GN*") }

# Initialize GN ID counter
$CurrentID = $StartID

foreach ($Subdir in $Subdirs) {
    # Get all files directly inside this subdir
    $Files = Get-ChildItem -Path $Subdir.FullName | Where-Object { -not $_.PSIsContainer }

    if ($Files.Count -gt 0) {
        # Format GN ID for this folder
        $ANIMAL_FOLDER = "GN$CurrentID"

        # Build the target path for this GN folder
        $TargetPath = Join-Path -Path $BaseDir -ChildPath "$ANIMAL_FOLDER\$ExperimentDate\$ExperimentName\$SessionNumber"

        # Create the directory structure
        New-Item -Path $TargetPath -ItemType Directory -Force | Out-Null

        # Move all files to the new folder
        foreach ($File in $Files) {
            Move-Item -Path $File.FullName -Destination $TargetPath
        }

        # Optionally, remove the original folder if it's now empty
        # Remove-Item -Path $Subdir.FullName -Force -Recurse

        # Increment GN ID for the next folder
        $CurrentID++
    }
}

Write-Host "Files moved successfully."
