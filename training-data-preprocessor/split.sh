#!/bin/bash

# Remove whitespaces from input files
for file in /input/* ; do
  mv "$file" "$(echo "$file" | sed -z 's/ //g')"
done

# Run spleeter on every input file
# shellcheck disable=SC2046
spleeter separate -p spleeter:4stems -o /output $(find /input -type f | sed -z 's/\n/ /g')

#
#for directory in /output/* ; do
#  # shellcheck disable=SC2164
#  cd "$directory"
#  ffmpeg -i bass.wav -i other.wav -i vocals.wav -filter_complex amerge=inputs=3 -ac 3 no-drums.wav
#  ffmpeg -i no-drums.wav -af aformat=s16:44100 no-drums.flac
#  ffmpeg -i drums.wav -af aformat=s16:44100 drums.flac
#  rm bass.wav other.wav vocals.wav drums.wav no-drums.wav
#done