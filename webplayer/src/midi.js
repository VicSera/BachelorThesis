const audioCtx = new AudioContext()

let notes = []
let nextContent = undefined
let nextNotes = []
let lastTime = 0

function setNextContent(data, play=false) {
    nextContent = 'data:audio/mid;base64,' + data.base64
    nextNotes = data.notes

    if (play) {
        playNext()
    }
}

function playNext() {
    notes = nextNotes
    startLoading()
    MIDI.Player.loadFile(nextContent, () => {
        nextNote = 0
        MIDI.Player.start()
        finishLoading()
        console.log("Playing next: ", nextContent)
        fetchNextContent()
    });
}

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

    setInterval(() => {
            if (MIDI.Player.playing && nextContent && Math.abs(MIDI.Player.currentTime - MIDI.Player.endTime) < 100)
                playNext()
        }, 1000
    )
}

function toBinary(string) {
  const codeUnits = new Uint16Array(string.length);
  for (let i = 0; i < codeUnits.length; i++) {
    codeUnits[i] = string.charCodeAt(i);
  }
  return btoa(String.fromCharCode(...new Uint8Array(codeUnits.buffer)));
}
//
// function process_midi(data) {
//     const content = 'data:audio/mid;base64,' + data.base64
//     notes = data.notes
//     MIDI.Player.loadFile(content, () => {
//         MIDI.Player.start()
//     });
// }