#!/bin/bash
# Prerequisite to make this script executable: chmod +x move_files_to_subdirs.sh
# Usage: ./move_files_to_subdirs.sh 25001 241210

# Input arguments
START_ID=$1
EXPERIMENT_DATE=$2

# Hardcoded parameters
EXPERIMENT_NAME="choices"
SESSION_NUM="session1"
# Base directory (update this if mounted differently on macOS)
BASE_DIR="/Volumes/DATA/experiment_trackball_Optomotor/locustVR"

# Input validation
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_id> <experiment_date>"
    exit 1
fi

# GN ID counter
ID=$START_ID

# Loop through subfolders (non-GN*) in BASE_DIR
for SUBDIR in "$BASE_DIR"/*; do
    if [ -d "$SUBDIR" ] && [[ "$(basename "$SUBDIR")" != GN* ]]; then
        # Find files directly inside the subdir
        FILES=("$SUBDIR"/*)
        FILE_COUNT=0

        for FILE in "${FILES[@]}"; do
            if [ -f "$FILE" ]; then
                ((FILE_COUNT++))
            fi
        done

        if [ "$FILE_COUNT" -gt 0 ]; then
            # Create GN ID folder
            ANIMAL_FOLDER="GN$ID"
            TARGET_DIR="$BASE_DIR/$ANIMAL_FOLDER/$EXPERIMENT_DATE/$EXPERIMENT_NAME/$SESSION_NUM"
            mkdir -p "$TARGET_DIR"

            # Move all files into new GN folder
            for FILE in "${FILES[@]}"; do
                if [ -f "$FILE" ]; then
                    BASENAME=$(basename "$FILE")
                    FILENAME="${BASENAME%.*}"
                    EXT="${BASENAME##*.}"

                    # If no extension, skip dot
                    if [ "$FILENAME" == "$EXT" ]; then
                        NEW_FILENAME="${FILENAME}_${OLD_FOLDER_NAME}"
                    else
                        NEW_FILENAME="${FILENAME}_${OLD_FOLDER_NAME}.${EXT}"
                    fi

                    mv "$FILE" "$TARGET_DIR/$NEW_FILENAME"
                fi
            done

            echo "Moved files from $(basename "$SUBDIR") to $ANIMAL_FOLDER"
            ((ID++))
        fi
    fi
done

echo "✅ All files moved."
