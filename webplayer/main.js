let exampleFile = undefined
let playPauseButton = undefined
let fileNameElement = undefined
let loadingSpinnerContainer = undefined
let sampleFileMessageElement = undefined

window.onload = function() {
    initGraphics()
    initMidi()

    playPauseButton = document.getElementById("playPauseButton")
    fileNameElement = document.getElementById("file-name")
    loadingSpinnerContainer = document.getElementById("spinner-container")
    sampleFileMessageElement = document.getElementById("example-file-message")
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

    playPauseButton.className = className
}

function refresh() {
    fetchNextContent(true, exampleFile, true)
    playPauseButton.className = "fa-solid fa-pause"
}

function fileUploaded() {
    exampleFile = document.getElementById("file-upload").files[0]
    fileNameElement.textContent = exampleFile.name
    sampleFileMessageElement.style.display = 'inline'
    refresh()
}

function finishLoading() {
    loadingSpinnerContainer.style.display = 'none'
}

function startLoading() {
    loadingSpinnerContainer.style.display = 'flex'
}