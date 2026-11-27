#!/usr/bin/env bash
# Rebuild showcase/ (poster.png, demo.gif, demo.mp4, showcase.json, latency_vs_n.png) from a clean clone.
#
# Requirements: python 3.9+, ffmpeg, vhs (https://github.com/charmbracelet/vhs, needs ttyd), curl.
#   python -m venv .venv && . .venv/bin/activate
#   pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
#   pip install -e ".[dev,serve]" -c requirements/constraints-cpu.txt
#
# Steps (all seeded, CPU only):
#   1. python -m rca_pipeline.demo_incident build        (Sock Shop bundle + incident, seed 42, ~2 min)
#      skipped when the committed bundle exists, unless FULL=1
#   2. python -m rca_pipeline.scaling_benchmark          (N=14..500, seeds 42,123,7; ~2-3 h on 2 cores)
#      skipped when results/scaling_benchmark.json exists, unless FULL=1
#   3. python showcase/figures.py                        (poster, latency chart, showcase.json)
#   4. vhs showcase/demo.tape against a live uvicorn     (terminal part of the demo)
#   5. ffmpeg: terminal recording + latency chart -> demo.mp4 (H.264) and demo.gif
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in python ffmpeg ffprobe vhs ttyd curl; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 1; }
done
export PYTHONHASHSEED=0
mkdir -p showcase/build

# Committed artifacts are reused (their SHA-256 is in the results manifest); FULL=1 retrains.
if [[ "${FULL:-0}" == 1 || ! -f examples/sockshop_demo/bundle/model.pt ]]; then
  python -m rca_pipeline.demo_incident build
fi
if [[ "${FULL:-0}" == 1 || ! -f results/scaling_benchmark.json ]]; then
  OMP_NUM_THREADS=2 python -m rca_pipeline.scaling_benchmark --train-threads 2 \
    --output-json results/scaling_benchmark.json
fi
python showcase/figures.py

MODEL_BUNDLE=examples/sockshop_demo/bundle python -m uvicorn serving.serve:app --port 8080 \
  --log-level warning > showcase/build/uvicorn.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -fs localhost:8080/ready >/dev/null && break; sleep 1; done
vhs showcase/demo.tape

# 8 s still of the measured latency chart appended to the terminal recording.
ffmpeg -y -loglevel error -i showcase/build/terminal.mp4 -loop 1 -t 8 -i showcase/latency_vs_n.png \
  -filter_complex "[0:v]scale=1280:720,fps=15,setsar=1[a];[1:v]scale=1280:720,fps=15,setsar=1,format=yuv420p[b];[a][b]concat=n=2:v=1[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart showcase/demo.mp4
ffmpeg -y -loglevel error -i showcase/demo.mp4 \
  -vf "fps=8,scale=1024:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  showcase/demo.gif
python - <<'PY'
import os, subprocess
d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "csv=p=0", "showcase/demo.mp4"]))
mb = os.path.getsize("showcase/demo.gif") / 1e6
print(f"demo.mp4 {d:.1f} s, demo.gif {mb:.1f} MB")
assert d <= 40 and mb < 10, "showcase limits exceeded (40 s, 10 MB)"
PY
echo "showcase rebuilt"
