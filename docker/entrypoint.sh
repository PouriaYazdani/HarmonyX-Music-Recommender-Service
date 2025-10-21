#!/usr/bin/env bash
set -euo pipefail


: "${DATA_DIR:=/app/data}"
: "${GH_OWNER:?Set GH_OWNER (e.g., PouriaYazdani)}"
: "${GH_REPO:?Set GH_REPO (e.g., HarmonyX-Music-Recommender-Service)}"
: "${GH_TAG:?Set GH_TAG (e.g., v0.1.0)}"
: "${ASSETS:=data.zip}"   # Comma-separated list; defaults to data.zip
: "${START_CMD:=uvicorn harmonyx.api:app --host 0.0.0.0 --port ${PORT:-8000}}"

mkdir -p "${DATA_DIR}"

base_url="https://github.com/${GH_OWNER}/${GH_REPO}/releases/download/${GH_TAG}"

download_asset() {
  local name="$1"
  local dest_path

  case "${name}" in
  *)
      dest_path="${DATA_DIR}/${name}"
      ;;
  esac


  if [ -f "${dest_path}" ]; then
    echo "Already present: ${dest_path}"
    return 0
  fi

  echo "Downloading ${name} → ${dest_path}"
  curl -L --fail --retry 3 -o "${dest_path}" "${base_url}/${name}"

  # Auto-extract common archives into DATA_DIR
  case "${dest_path}" in
    *.zip)
      echo "Extracting ${dest_path} to ${DATA_DIR}"
      unzip -o "${dest_path}" -d "${DATA_DIR}"
      ;;
    *.tar.gz|*.tgz)
      echo "Extracting ${dest_path} to ${DATA_DIR}"
      tar -xzf "${dest_path}" -C "${DATA_DIR}"
      ;;
  esac
}

IFS=',' read -ra arr <<< "${ASSETS}"
for a in "${arr[@]}"; do
  a_trimmed="$(echo "$a" | xargs)"
  [ -n "${a_trimmed}" ] && download_asset "${a_trimmed}"
done


echo "Data in ${DATA_DIR}:"
ls -lh "${DATA_DIR}" || true

echo "Starting app: ${START_CMD}"
exec ${START_CMD}
