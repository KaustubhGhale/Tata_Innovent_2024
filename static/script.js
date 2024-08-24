/* script.js */
document.getElementById('submit-button').addEventListener('click', function() {
    const fileInput = document.getElementById('file-upload');
    const selectedModel = document.getElementById('model-select').value;
    if (fileInput.files.length === 0) {
        alert('Please upload an image or video.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', selectedModel);

    // Fetch request to backend (Python) for processing
    fetch('/process', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Display results
        document.getElementById('results').style.display = 'block';
        document.getElementById('ai-response').textContent = data.aiResponse;
        document.getElementById('helpline').textContent = data.helpline;
        document.getElementById('generated-image').src = data.generatedImage;
    })
    .catch(error => console.error('Error:', error));
});
