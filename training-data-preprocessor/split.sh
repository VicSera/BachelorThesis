#!/bin/bash

# Remove whitespaces from input files
for file in /input/* ; do
  mv "$file" "$(echo "$file" | sed -z 's/ //g')"
done

# Run spleeter on every input file
# shellcheck disable=SC2046
# UNCOMMENT NEXT LINE
#spleeter separate -p spleeter:4stems -o /output $(find /input -type f | sed -z 's/\n/ /g')

#
for directory in /output/* ; do
  # shellcheck disable=SC2164
  cd "$directory"
  # UNCOMMENT NEXT 4 LINES
#  ffmpeg -i bass.wav -i other.wav -i vocals.wav -filter_complex amerge=inputs=3 -ac 3 no-drums.wav
#  ffmpeg -i no-drums.wav -af aformat=s16:44100 no-drums.flac
#  ffmpeg -i drums.wav -af aformat=s16:44100 drums.flac
#  rm bass.wav other.wav vocals.wav drums.wav no-drums.wav

  ffmpeg -i no-drums.flac -filter_complex "[0:v]scale=-2:720,format=yuv420p[v];[0:a]amerge=inputs=$(ffprobe -loglevel error -select_streams a -show_entries stream=codec_type -of csv=p=0 no-drums.flac | wc -l)[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 23 -preset medium -c:a libmp3lame -ar 44100 -ac 2 no-drums-single.flac
  ffmpeg -i drums.flac -filter_complex "[0:v]scale=-2:720,format=yuv420p[v];[0:a]amerge=inputs=$(ffprobe -loglevel error -select_streams a -show_entries stream=codec_type -of csv=p=0 drums.flac | wc -l)[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 23 -preset medium -c:a libmp3lame -ar 44100 -ac 2 drums-single.flac

#
#  ffmpeg -i no-drums.flac -c:v copy -c:a aac -b:a 160k -ac 2 -filter_complex amerge=inputs=2 no-drums-single.flac
#  ffmpeg -i drums.flac -c:v copy -c:a aac -b:a 160k -ac 2 -filter_complex amerge=inputs=2 drums-single.flac
done