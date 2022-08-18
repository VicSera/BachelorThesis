const params = {
    fitted: true
}
const notes = 128
const whiteKeyCount = 75
const keyHeight = 150
const keys = []
let two = undefined

function initGraphics() {
    const elem = document.body.getElementsByTagName('div')[1]
    two = new Two(params).appendTo(elem)


    const keyWidth = two.width / whiteKeyCount
    const y = two.height * 0.5
    var lastX = -keyWidth / 2

    const whiteKeyGroup = two.makeGroup()
    const blackKeyGroup = two.makeGroup()

    for (let i = 0; i < notes; i++) {
        const rank = i % 12

        // black key
        if (rank === 1 || rank === 4 || rank === 6 || rank === 9 || rank === 11) {
            const x = lastX + keyWidth / 2
            const rect = two.makeRectangle(x, y - keyHeight / 8, keyWidth * 3/4, keyHeight * 3/4)

            rect.fill = 'black'

            keys.push(rect)
            blackKeyGroup.add(rect)
        } else { // white key
            const x = lastX + keyWidth
            lastX = x
            const rect = two.makeRectangle(x, y, keyWidth, keyHeight)

            rect.fill = 'white'
            rect.stroke = 'black'
            rect.linewidth = 3;

            keys.push(rect)
            whiteKeyGroup.add(rect)
        }
    }

    two.update();
}

function highlightKeys(highlighted) {
    for (let idx = 0; idx < notes; idx++) {
        const rank = idx % 12
        if (highlighted.find(i => i === idx)) {
            keys[idx].fill = 'red'
        } else {
            if (rank === 1 || rank === 4 || rank === 6 || rank === 9 || rank === 11) {
                keys[idx].fill = 'black'
            } else {
                keys[idx].fill = 'white'
            }
        }
    }
    two.update()
}