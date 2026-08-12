Output assets/demo-qwen.gif

Require uv

Set Shell "bash"
Set Width 760
Set Height 900
Set FontSize 16
Set TypingSpeed 1ms

# Launch completely off-camera

Hide

Type "MODEL=${NATWALK_QWEN_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/snapshots/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q6_K.gguf}"
Enter

Type "uv run --with-editable . examples/llm_app.py 'The most interesting consequence of information theory is' --model-path $MODEL"
Enter

# Don't guess how long model loading takes.
# This appears only once Natwalk owns the fullscreen terminal.

Wait+Screen /suggestion/

# Also let background search fill in a nice tree.
# Hidden time contributes nothing to the GIF.

Sleep 4s

Show

# Establish the scene

Sleep 1500ms

# The money shot: sprint down known probability structure

Right@100ms 7
Sleep 900ms

# Show that navigation is genuinely reversible

Left@120ms 3
Sleep 900ms

# Pick a different continuation

Down@300ms 2
Sleep 700ms

# Dive into that branch

Right@120ms 4
Sleep 1800ms

# End on the beautiful frame rather than showing the shell again

Hide
Type "q"
