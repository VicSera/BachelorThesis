const audioCtx = new AudioContext()

let notes = []

function initMidi() {
    MIDI.loadPlugin({
        soundfontUrl: "lib/soundfont/",
        instrument: "acoustic_grand_piano",
        onsuccess: function() {
            console.log('MIDI booted up')
        }
    })

    MIDI.Player.clearAnimation();
    MIDI.Player.setAnimation(function (data) {
        if (MIDI.Player.playing) {
            const highlighted = Object.keys(data.events).map(str => parseInt(str))
            renderFallingNotes(data.now)
            highlightKeys(highlighted)
            two.update()
        }
    })
}

function toBinary(string) {
  const codeUnits = new Uint16Array(string.length);
  for (let i = 0; i < codeUnits.length; i++) {
    codeUnits[i] = string.charCodeAt(i);
  }
  return btoa(String.fromCharCode(...new Uint8Array(codeUnits.buffer)));
}

function process_midi(data) {
    const content = 'data:audio/mid;base64,' + data.base64
    notes = data.notes
    MIDI.Player.loadFile(content, () => {
        MIDI.Player.start()
    });
}