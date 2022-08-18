function fetch_midi() {
    audioCtx.resume()

    const formData = new FormData()
    formData.append('length', '100')

    axios.post('http://localhost:5000/generate', formData)
        .then(function (response) {
            process_midi(response.data)
        })
        .catch(function (error) {
            alert('Something went wrong! - ' + error)
        })
}