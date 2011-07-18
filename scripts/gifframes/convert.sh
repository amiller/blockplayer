for f in *.gif; do mkdir -p fireworks.gif_frames/; convert +adjoin -coalesce fireworks.gif fireworks.gif_frames/frame%02d.gif; done
