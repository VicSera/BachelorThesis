#!/bin/bash

# Run spleeter on every input file
# shellcheck disable=SC2046
spleeter separate -p spleeter:4stems -o /output $(find /input -type f | sed -z 's/\n/ /g')

#
for directory in /output/* ; do
  # shellcheck disable=SC2164
  cd "$directory"
  ffmpeg -i bass.wav -i other.wav -i vocals.wav -filter_complex amerge=inputs=3 -ac 3 no-drums.wav
  rm bass.wav other.wav vocals.wav
done