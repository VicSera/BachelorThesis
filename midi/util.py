from mido import MidiFile

mid = MidiFile('../data/dataset/groove/drummer1/session1/1_funk_80_beat_4-4.mid')

for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)