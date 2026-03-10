#!/usr/bin/env bash
set -euo pipefail

# Download all files from a Figshare collection (by collection id) using the public API.
# Avoids scraping springernature.figshare.com item pages (WAF/JS).
#
# Requirements: curl, jq

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "Error: missing dependency: $1" >&2; exit 1; }
}

require curl
require jq

COLLECTION_ID="${1:-6829134}"
OUTDIR="${OUTDIR:-figshare_collection_${COLLECTION_ID}}"
API_BASE="${API_BASE:-https://api.figshare.com/v2}"
PAGE_SIZE="${PAGE_SIZE:-100}"

mkdir -p "$OUTDIR"

sanitize_filename() {
  local name="$1"
  name="${name//$'\n'/ }"
  name="${name//$'\r'/ }"
  name="${name//\//_}"
  name="${name//\\/ _}"
  name="${name//:/-}"
  name="${name//\*/_}"
  name="${name//\?/_}"
  name="${name//\"/_}"
  name="${name//</_}"
  name="${name//>/_}"
  name="${name//|/_}"
  printf '%s' "$name"
}

download_url_to_file() {
  local url="$1"
  local dest="$2"

  if [[ -f "$dest" ]]; then
    echo "Skip (exists): $(basename "$dest")"
    return 0
  fi

  # Important: do GET (not HEAD). Some backends deny HEAD.
  # Retry the whole request so short-lived redirects/presigned URLs refresh.
  local attempts=8
  local i=1
  while [[ $i -le $attempts ]]; do
    echo "Download: $(basename "$dest") (attempt $i/$attempts)"
    if curl -fL --connect-timeout 20 --max-time 600 -o "$dest.part" "$url"; then
      if [[ -s "$dest.part" ]]; then
        mv "$dest.part" "$dest"
        return 0
      fi
    fi
    rm -f "$dest.part"
    sleep 1
    i=$((i + 1))
  done

  echo "Error: failed to download after $attempts attempts: $url" >&2
  return 1
}

echo "Collection: ${COLLECTION_ID}"
echo "API base:   ${API_BASE}"
echo "Outdir:     ${OUTDIR}"
echo

page=1
total_articles=0

while :; do
  articles_json="$(
    curl -fsSL \
      "${API_BASE}/collections/${COLLECTION_ID}/articles?page=${page}&page_size=${PAGE_SIZE}"
  )"

  count="$(jq 'length' <<<"$articles_json")"
  if [[ "$count" -eq 0 ]]; then
    break
  fi

  echo "Page ${page}: ${count} articles"
  total_articles=$((total_articles + count))

  # Each element is an "article presenter" with an id
  jq -r '.[].id' <<<"$articles_json" | while read -r article_id; do
    article_json="$(curl -fsSL "${API_BASE}/articles/${article_id}")"

    article_title="$(jq -r '.title // ("article_" + (.id|tostring))' <<<"$article_json")"
    safe_title="$(sanitize_filename "$article_title")"
    article_dir="${OUTDIR}/${article_id} - ${safe_title}"
    mkdir -p "$article_dir"

    # For each file: use API-provided name + download_url
    jq -r '.files[] | [.name, .download_url] | @tsv' <<<"$article_json" \
      | while IFS=$'\t' read -r raw_name download_url; do
          name="$(sanitize_filename "$raw_name")"
          dest="${article_dir}/${name}"
          download_url_to_file "$download_url" "$dest"
        done
  done

  page=$((page + 1))
done

echo
echo "Done. Downloaded files for ${total_articles} articles into: ${OUTDIR}"
echo "\nCleaning up..."
find . -mindepth 2 -type f -exec mv -t . -n -- {} +
find . -type d -empty -delete
