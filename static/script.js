// script.js

document.getElementById('submit-button').addEventListener('click', function() {
    processImage();
});

document.getElementById('voice-command').addEventListener('click', function() {
    startVoiceRecognition();
});

function processImage() {
    const fileInput = document.getElementById('file-upload');
    const selectedModel = document.getElementById('model-select').value;
    if (fileInput.files.length === 0) {
        alert('Please upload an image.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', selectedModel);

    fetch('/process', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data);
    })
    .catch(error => console.error('Error:', error));
}

function displayResults(data) {
    document.getElementById('results').style.display = 'block';
    document.getElementById('ai-response').textContent = data.aiResponse;
    document.getElementById('helpline').textContent = data.helpline;
    document.getElementById('generated-image').src = data.image_url;

    // Use text-to-speech to read out the AI response and helpline
    speak(data.aiResponse + '. ' + data.helpline);
}

function startVoiceRecognition() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.start();

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript.toLowerCase();
        handleVoiceCommand(transcript);
    };

    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
        alert('Sorry, I did not catch that. Please try again.');
    };
}

function handleVoiceCommand(command) {
    if (command.includes('upload image')) {
        document.getElementById('file-upload').click();
    } else if (command.includes('submit')) {
        processImage();
    } else if (command.includes('choose model one')) {
        document.getElementById('model-select').value = 'model1';
    } else {
        alert('Command not recognized.');
    }
}

function speak(text) {
    const synthesis = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    synthesis.speak(utterance);
}
