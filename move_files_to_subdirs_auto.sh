
#!/bin/bash
# Prerequisite: chmod +x move_files_to_subdirs.sh
# Usage: ./move_files_to_subdirs.sh 25001

# Input argument
START_ID=$1

# Hardcoded parameters
EXPERIMENT_NAME="choices"
SESSION_NUM="session1"
BASE_DIR="/Users/aljoscha/Downloads/Data-2"

# Input validation
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <start_id>"
    exit 1
fi

# GN ID counter
ID=$START_ID

# Loop through subfolders (non-GN*) in BASE_DIR
for SUBDIR in "$BASE_DIR"/*; do
    if [ -d "$SUBDIR" ] && [[ "$(basename "$SUBDIR")" != GN* ]]; then
        BASENAME=$(basename "$SUBDIR")

        # Extract date (expects format like 20250627 at beginning of folder name)
        if [[ "$BASENAME" =~ ^([0-9]{8}) ]]; then
            EXPERIMENT_DATE="${BASH_REMATCH[1]}"
        else
            echo "❌ Could not extract date from folder name: $BASENAME"
            continue
        fi

        # Count files directly in subdir
        FILES=("$SUBDIR"/*)
        FILE_COUNT=0

        for FILE in "${FILES[@]}"; do
            if [ -f "$FILE" ]; then
                ((FILE_COUNT++))
            fi
        done

        if [ "$FILE_COUNT" -gt 0 ]; then
            ANIMAL_FOLDER="GN$ID"
            TARGET_DIR="$BASE_DIR/$ANIMAL_FOLDER/$EXPERIMENT_DATE/$EXPERIMENT_NAME/$SESSION_NUM"
            mkdir -p "$TARGET_DIR"

            # Move and rename files
            for FILE in "${FILES[@]}"; do
                if [ -f "$FILE" ]; then
                    FILE_NAME=$(basename "$FILE")
                    NAME_NO_EXT="${FILE_NAME%.*}"
                    EXT="${FILE_NAME##*.}"

                    if [ "$NAME_NO_EXT" == "$EXT" ]; then
                        NEW_FILENAME="${NAME_NO_EXT}_${BASENAME}"
                    else
                        NEW_FILENAME="${NAME_NO_EXT}_${BASENAME}.${EXT}"
                    fi

                    mv "$FILE" "$TARGET_DIR/$NEW_FILENAME"
                fi
            done

            echo "✅ Moved files from $BASENAME to $ANIMAL_FOLDER"
            ((ID++))
        fi
    fi
done

echo "🎉 All files moved."