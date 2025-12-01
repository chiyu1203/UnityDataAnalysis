# Usage in PowerShell as admin:
# map network again by the following command. net use Z: "\\files.uni-konstanz.de\AG_Couzin-Fuchs"
# Then move to the directory where this script is located and run:
# .\move_files_to_subdirs.ps1 -StartID 25001 -ExperimentDate 250624
param (
    [Parameter(Mandatory=$true)][int]$StartID,
    [Parameter(Mandatory=$true)][string]$ExperimentDate
)

# Hardcoded parameters
$ExperimentName = "collision"
$SessionNumber = "session1"
# Base directory
$BaseDir = "Z:\DATA\experiment_trackball_Optomotor\locustVR"

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
        $OldFolderName = $Subdir.Name

        # Move all files to the new folder
        foreach ($File in $Files) {
            $OriginalName = [System.IO.Path]::GetFileNameWithoutExtension($File.Name)
            $Extension = [System.IO.Path]::GetExtension($File.Name)

            # Build new filename with old folder name appended
            if ([string]::IsNullOrWhiteSpace($Extension)) {
                $NewFileName = "${OriginalName}_$OldFolderName"
            } else {
                $NewFileName = "${OriginalName}_$OldFolderName$Extension"
            }

            $DestinationPath = Join-Path -Path $TargetPath -ChildPath $NewFileName
            Move-Item -Path $File.FullName -Destination $DestinationPath
        }

        Write-Host "Moved files from '$OldFolderName' to '$ANIMAL_FOLDER'"
        $CurrentID++
    }
}

Write-Host "Files moved successfully."
