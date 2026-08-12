Output assets/demo-muscriptor.gif

Require uv

Set Shell "bash"
Set Width 760
Set Height 900
Set FontSize 16
Set TypingSpeed 1ms

Hide

Type "uv run --with-editable . examples/muscriptor_app.py '../muscriptor/samples/Laura Marling - What He Wrote.mp3' --model medium --device cuda --chunk 0"
Enter

Wait+Screen /suggestion/

# Give search some hidden time to populate the tree.
Sleep 4s

Show
Sleep 1500ms

Right@120ms 6
Sleep 900ms

Left@140ms 3
Sleep 800ms

Down@300ms 2
Sleep 700ms

Right@120ms 4
Sleep 1800ms

Hide
Type "q"
