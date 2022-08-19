function fetchNextContent(play=false) {
    audioCtx.resume()

    const formData = new FormData()
    formData.append('length', '50')

    axios.post('http://localhost:5000/generate', formData)
        .then(function (response) {
            setNextContent(response.data, play)
        })
        .catch(function (error) {
            alert('Something went wrong! - ' + error)
        })
}