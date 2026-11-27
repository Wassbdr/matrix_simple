# Rebuild the public incident demo

From the checkout, activate the Python environment, then install the project with `make install INSTALL_TARGET='.[data]'`. Install FFmpeg (including ffprobe), [VHS](https://github.com/charmbracelet/vhs), and [ttyd](https://github.com/tsl0922/ttyd) for recording.

- `make demo` runs the installed `rca rank` on an RE2 Online Boutique case. If the Figshare release is not already local, `rca data --demo` fetches and verifies only the metrics and timestamp from the [pinned public mirror](../docs/rcaeval_data.md).
- `make showcase` records the CLI and regenerates the poster, latency chart, GIF, MP4, and JSON. It uses the packaged pooled RE1 model, trained with seed 42. It does not start the HTTP service or require the full data archive.
- The BARO output is the cached prediction from the official reference, not a local approximation. Reproduce it with `make bench` using the [reference setup](../docs/rcaeval_data.md). Use `make scaling` to regenerate the measured CPU chart data, then rerun `make showcase`.
