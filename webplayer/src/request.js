function fetchNextContent(play=false, exampleFile = undefined, showLoading = false) {
    audioCtx.resume()

    const formData = new FormData()
    formData.append('length', '50')
    if (exampleFile != null) {
        formData.append('exampleFile', exampleFile)
    }

    showLoading && startLoading()
    axios.post('http://localhost:5000/generate', formData)
        .then(function (response) {
            setNextContent(response.data, play)
            showLoading && finishLoading()
        })
        .catch(function (error) {
            alert('Something went wrong! - ' + error)
        })
}