#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
if [ -f "$SCRIPT_DIR/.settings" ]; then
  source "$SCRIPT_DIR/.settings"
fi
API_TOKEN="${WIKI_API_TOKEN}"
BASE_URL="${WIKI_BASE_URL:-https://wiki.kiezburn.org/api}"
COLLECTION_ID="${WIKI_COLLECTION_ID_2026}"
OUTPUT_DIR="$(dirname "$0")/wiki_pages"

mkdir -p "$OUTPUT_DIR"

# Fetch all document IDs with pagination
echo "Fetching document list..."
ALL_DOCS='[]'
OFFSET=0
LIMIT=100

while true; do
  RESPONSE=$(curl -s "$BASE_URL/documents.list" -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_TOKEN" \
    -d "{\"collectionId\": \"$COLLECTION_ID\", \"limit\": $LIMIT, \"offset\": $OFFSET}")

  COUNT=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['data']))")
  TOTAL=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['pagination']['total'])")

  echo "Fetched $COUNT docs (offset $OFFSET / total $TOTAL)"

  # Extract and download each doc
  echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for doc in data['data']:
    print(doc['id'] + '|' + doc['title'])
" | while IFS='|' read -r DOC_ID DOC_TITLE; do
    # Sanitize filename
    SAFE_TITLE=$(echo "$DOC_TITLE" | tr '/:*?"<>|\\' '_' | tr -s ' ' '_' | cut -c1-100)
    FILENAME="$OUTPUT_DIR/${SAFE_TITLE:-untitled}_${DOC_ID:0:8}.md"

    echo "  Downloading: $DOC_TITLE"
    curl -s "$BASE_URL/documents.export" -X POST \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $API_TOKEN" \
      -d "{\"id\": \"$DOC_ID\"}" \
      | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data.get('ok'):
    print(data['data'])
else:
    print('ERROR: ' + str(data))
" > "$FILENAME"
  done

  if [ "$((OFFSET + LIMIT))" -ge "$TOTAL" ]; then
    break
  fi
  OFFSET=$((OFFSET + LIMIT))
done

echo "Done! Files saved to $OUTPUT_DIR"
ls "$OUTPUT_DIR" | wc -l
