# Rendered by showcase/make_showcase.sh (needs the STA service on :8080).
Output showcase/build/terminal.mp4
Set Shell "bash"
Set FontSize 18
Set Width 1280
Set Height 720
Set Padding 24
Set Theme "GitHub Dark"
Set TypingSpeed 35ms

Hide
Type "source .venv/bin/activate 2>/dev/null; export PS1='$ '; clear"
Enter
Show

Type "# Live: FastAPI STA service on :8080, model bundle examples/sockshop_demo/bundle"
Enter
Sleep 500ms
Type "python scripts/demo_predict.py   # POST a synthetic Sock Shop incident (seed 42) to /predict"
Enter
Sleep 8s
Hide
Type "clear"
Enter
Show
Type "python -m rca_pipeline.demo_incident run   # same incident: STA bundle vs BARO, same candidates"
Enter
Sleep 8s
