Output showcase/build/terminal.mp4
Set Shell "bash"
Set FontSize 22
Set Width 1280
Set Height 720
Set Padding 36
Set Theme "GitHub Dark"
Set TypingSpeed 45ms

Hide
Type "export PS1='$ '; clear"
Enter
Show
Type "# RCAEval Online Boutique: checkoutservice CPU fault"
Enter
Sleep 1s
Type `rca rank "$SHOWCASE_CASE" --top-k 3`
Enter
Sleep 5s
Type "python showcase/reference.py"
Enter
Sleep 8s
