const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const answerBox = document.querySelector('.answer-box');

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'user-message';
        userMessageDiv.textContent = message;
        answerBox.appendChild(userMessageDiv);
        userInput.value = '';
        answerBox.scrollTop = answerBox.scrollHeight;

        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'bot-message';
        answerBox.appendChild(botMessageDiv);
        answerBox.scrollTop = answerBox.scrollHeight;

        let dotCount = 1;
        const typingText = "CuesBot is typing";
        botMessageDiv.textContent = typingText + ".";
        const typingInterval = setInterval(() => {
            dotCount = dotCount < 4 ? dotCount + 1 : 1;
            botMessageDiv.textContent = typingText + ".".repeat(dotCount);
        }, 500);

        try {
            const response = await fetch("http://127.0.0.1:8000/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: message,
                    model: "mistral"
                })
            });

            const data = await response.json();
            clearInterval(typingInterval); 
            botMessageDiv.textContent = (data.answer || "Sorry, no response received.");
            answerBox.scrollTop = answerBox.scrollHeight;

        } catch (error) {
            clearInterval(typingInterval);
            botMessageDiv.textContent = "Sorry, I am not able to connect to the server.";
            console.error(error);
        }
    }

    sendBtn.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });