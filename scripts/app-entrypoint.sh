#!/bin/bash
set -e

# Seed family.md into the named volume on first run.
# On subsequent runs the file already exists and this is a no-op.
if [ ! -f "$FAMILY_MD_PATH" ]; then
    echo "Seeding family.md to $FAMILY_MD_PATH"
    cp /app/family.md.default "$FAMILY_MD_PATH"
fi

exec "$@"
