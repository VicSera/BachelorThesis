const params = {
    fitted: true
}
const noteCount = 128
const whiteKeyCount = 75
const keyHeight = 150
const keys = []
let fallingNotes = []
const windowSizeInSec = 2
let two = undefined
let height = undefined
let previewHeight = undefined

let nextNote = 0

let noteGroup

function initGraphics() {
    const elem = document.body.getElementsByTagName('div')[1]
    two = new Two(params).appendTo(elem)


    const keyWidth = two.width / whiteKeyCount
    height = two.height
    previewHeight = height - keyHeight
    var lastX = -keyWidth / 2

    noteGroup = two.makeGroup()
    const whiteKeyGroup = two.makeGroup()
    const blackKeyGroup = two.makeGroup()

    for (let i = 0; i < noteCount; i++) {
        const rank = i % 12

        // black key
        if (rank === 1 || rank === 4 || rank === 6 || rank === 9 || rank === 11) {
            const x = lastX + keyWidth / 2
            const rect = two.makeRectangle(x, height - keyHeight * 5/8, keyWidth * 3/4, keyHeight * 3/4)

            rect.fill = 'black'

            keys.push(rect)
            blackKeyGroup.add(rect)
        } else { // white key
            const x = lastX + keyWidth
            lastX = x
            const rect = two.makeRectangle(x, height - keyHeight / 2, keyWidth, keyHeight)

            rect.fill = 'white'
            rect.stroke = 'black'
            rect.linewidth = 2;

            keys.push(rect)
            whiteKeyGroup.add(rect)
        }
    }

    two.update();
}

function highlightKeys(highlighted) {
    for (let idx = 0; idx < noteCount; idx++) {
        const rank = idx % 12
        if (highlighted.find(i => i === idx)) {
            keys[idx].fill = 'yellow'
        } else {
            if (rank === 1 || rank === 4 || rank === 6 || rank === 9 || rank === 11) {
                keys[idx].fill = 'black'
            } else {
                keys[idx].fill = 'white'
            }
        }
    }
}

const getDuration = (note) => note.end - note.start

const getNoteHeight = (note) => getDuration(note) / windowSizeInSec * previewHeight

function calculateNoteY(note, now) {
    const height = getNoteHeight(note)
    const startsIn = note.start - now
    return previewHeight - (startsIn / windowSizeInSec) * previewHeight - height / 2
}

function renderFallingNotes(now) {
    // Remove finished notes
    while (fallingNotes.length > 0 && fallingNotes[0].note.end < now) {
        fallingNotes[0].rect.remove();
        fallingNotes.shift()
    }

    // Add next notes
    while (nextNote < notes.length && notes[nextNote].start < now + windowSizeInSec) {
        const note = notes[nextNote]
        const key = keys[note.pitch]

        const rect = two.makeRectangle(key.position.x,0, key.width, getNoteHeight(note))
        rect.fill = 'red'
        fallingNotes.push({
            note: note,
            rect: rect
        })
        nextNote++

        noteGroup.add(rect)
        two.update()
    }

    // Update y coordinates
    for (let i = 0; i < fallingNotes.length; i++) {
        const rect = fallingNotes[i].rect

        rect.position.y = calculateNoteY(fallingNotes[i].note, now)
    }
}