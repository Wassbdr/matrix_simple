#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in python rca ffmpeg ffprobe vhs ttyd; do
  command -v "$tool" >/dev/null || { echo "Install $tool; see showcase/README.md." >&2; exit 1; }
done
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SHOWCASE_CASE=$(rca data --demo --root "${RCA_DATA:-$HOME/data/rcaeval}")
export SHOWCASE_CASE
mkdir -p showcase/build
rca rank "$SHOWCASE_CASE" --top-k 1000 --json > showcase/build/sta.json
python showcase/figures.py
vhs showcase/demo.tape

ffmpeg -y -loglevel error -filter_complex_threads 1 -i showcase/build/terminal.mp4 \
  -loop 1 -t 8 -i showcase/latency_vs_n.png \
  -filter_complex "[0:v]scale=1280:720,fps=15,setsar=1,format=yuv420p[a];[1:v]scale=1280:720,fps=15,setsar=1,format=yuv420p[b];[a][b]concat=n=2:v=1[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart showcase/demo.mp4
ffmpeg -y -loglevel error -filter_threads 1 -i showcase/demo.mp4 \
  -vf "fps=8,scale=1024:-1,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  showcase/demo.gif
python - <<'PY'
import json
import subprocess
from pathlib import Path

def probe(filename):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", filename,
    ]))

gif = probe("showcase/demo.gif")
width = gif["streams"][0]["width"]
duration = float(gif["format"]["duration"])
size = Path("showcase/demo.gif").stat().st_size
assert duration <= 40 and width <= 1280 and size < 10_000_000, gif
mp4 = probe("showcase/demo.mp4")["streams"][0]
assert mp4["codec_name"] == "h264", mp4
poster = probe("showcase/poster.png")["streams"][0]
assert (poster["width"], poster["height"]) == (1280, 720), poster
print(f"demo.gif: {duration:.2f} s, {width} px wide, {size} bytes; demo.mp4: H.264; poster: 1280x720")
PY
