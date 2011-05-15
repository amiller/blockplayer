# Convert videos from ogv to XVID avi 
for x in *.ogv; do ffmpeg -i $x -vcodec libxvid -b 18000k `echo "${x}" | sed -e "s/.ogv/.avi/"`; done

# Convert videos from MTS to XVID avi
for x in *.MTS; do ffmpeg -i $x -vcodec libxvid -b 18000k `echo "${x}" | sed -e "s/.MTS/.avi/"`; done
