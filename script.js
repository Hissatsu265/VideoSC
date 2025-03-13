document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const youtubeForm = document.getElementById('youtube-form');
    const youtubeUrlInput = document.getElementById('youtube-url');
    const resultSection = document.getElementById('result-section');
    const videoEmbed = document.getElementById('video-embed');
    const videoTitle = document.getElementById('video-title');
    const videoChannel = document.getElementById('video-channel');
    const summaryText = document.getElementById('summary-text');
    const loadingSpinner = document.querySelector('.loading-spinner');
    const chatToggleBtn = document.getElementById('chat-toggle-btn');
    const chatSection = document.getElementById('chat-section');
    const chatContainer = document.getElementById('chat-container');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');

    // Mock data for testing
    const mockSummary = "Video này hiện không có phụ đề cho tiếng anh. Hãy thử xem video và đặt câu hỏi cho tôi!";
    
    const youtubeRegex = /^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})(?:\S+)?$/;

    // Form submission
    youtubeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const youtubeUrl = youtubeUrlInput.value.trim();
        const match = youtubeUrl.match(youtubeRegex);
        
        if (match) {
            const videoId = match[1];
            processVideo(videoId);
        } else {
            alert('Vui lòng nhập một URL YouTube hợp lệ');
        }
    });

    function processVideo(videoId) {
        resultSection.classList.remove('hidden');
        loadingSpinner.classList.remove('hidden');
        summaryText.innerHTML = '';
        chatSection.classList.add('hidden');
        
        // Embed video
        videoEmbed.innerHTML = `<iframe src="https://www.youtube.com/embed/${videoId}" allowfullscreen></iframe>`;
        
        videoTitle.textContent = "Video về phương pháp học tập hiệu quả";
        videoChannel.textContent = "Kênh Học Tập";
        
        // Simulate API call for summary
        setTimeout(() => {
            loadingSpinner.classList.add('hidden');
            summaryText.textContent = mockSummary;
        }, 2000);
        
        // Reset chat container
        chatContainer.innerHTML = `
            <div class="chat-message system-message">
                <div class="message-content">
                    Xin chào! Bạn có thể hỏi tôi bất kỳ câu hỏi nào về nội dung của video này.
                </div>
            </div>
        `;
    }

    // Toggle chat section
    chatToggleBtn.addEventListener('click', function() {
        chatSection.classList.toggle('hidden');
        if (!chatSection.classList.contains('hidden')) {
            chatInput.focus();
        }
    });

    // Handle chat form submission
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const question = chatInput.value.trim();
        if (!question) return;
        
        // Add user message to chat
        addMessage(question, 'user');
        chatInput.value = '';
        
        // Simulate API call for chat response
        setTimeout(() => {
            // Try to call backend API (simulated)
            try {
                // This would be a real API call in the complete app
                // For now, simulate a random chance of error
                if (Math.random() < 0.3) {
                    throw new Error('Simulated backend error');
                }
                
                // Generate a somewhat relevant response based on the question and the mock summary
                const response = generateMockResponse(question);
                addMessage(response, 'system');
            } catch (error) {
                // If error, just respond with "Sorry"
                addMessage("Sorry", 'system');
            }
        }, 1000);
    });

    // Add message to chat container
    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageDiv.appendChild(messageContent);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function generateMockResponse(question) {
        const lowerQuestion = question.toLowerCase();
        
      
    }

    // Extract video ID from YouTube URL
    function extractVideoId(url) {
        const match = url.match(youtubeRegex);
        return match ? match[1] : null;
    }
});