window.onload = function() {
    initGraphics()
    initMidi()
}

function togglePlay() {
    let className
    if (MIDI.Player.playing) {
        MIDI.Player.pause()
        className = "fa-solid fa-play"
    } else {
        MIDI.Player.resume()
        className = "fa-solid fa-pause"
    }

    document.getElementById("playPauseButton").className = className
}

function refresh() {
    fetchNextContent(true)
    document.getElementById("playPauseButton").className = "fa-solid fa-pause"
}