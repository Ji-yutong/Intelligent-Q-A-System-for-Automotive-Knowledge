document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    function appendMessage(text, isUser) {
        const bubble = document.createElement('div');
        bubble.className = 'bubble ' + (isUser ? 'user-bubble' : 'model-bubble');
        bubble.textContent = text;
        chatBox.appendChild(bubble);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function sendMessage() {
        const message = userInput.value;
        if (message.trim() === '') return;

        appendMessage(message, true);
        userInput.value = '';

        fetch('/send_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_input: message })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage(data.response, false);
        })
        .catch(error => console.error('Error:', error));
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendMessage();
        }
    });

    // Load dialog history on page load
    fetch('/get_dialog_history')
        .then(response => response.json())
        .then(data => {
            data.forEach(entry => {
                appendMessage(entry.response || entry.user_input, !!entry.response);
            });
        });
});
