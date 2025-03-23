// Global variables
let chatHistory = [];
let currentChatId = null;
let isTyping = false;
let uploadedFile = null;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize elements
    const sendButton = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const newChatButton = document.getElementById('new-chat-btn');
    const chatModeSelect = document.getElementById('chat-mode');
    const chatHistoryContainer = document.getElementById('chat-history');
    const fileUploadContainer = document.getElementById('file-upload-container');
    const reportFileInput = document.getElementById('report-file');
    
    // Load marked library for Markdown parsing
    loadMarkdownLibrary();
    
    // Load chat history from local storage
    loadChatHistory();
    
    // Create a new chat if none exists
    if (!currentChatId) {
        createNewChat();
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto resize the textarea based on content
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        // Limit height to 120px max
        if (this.scrollHeight > 120) {
            this.style.height = '120px';
            this.style.overflowY = 'auto';
        } else {
            this.style.overflowY = 'hidden';
        }
    });
    
    newChatButton.addEventListener('click', createNewChat);
    
    chatModeSelect.addEventListener('change', function() {
        // Update current chat mode
        if (currentChatId) {
            const currentChat = chatHistory.find(chat => chat.id === currentChatId);
            if (currentChat) {
                currentChat.mode = chatModeSelect.value;
                saveChatsToLocalStorage();
                
                // Show/hide file upload based on mode
                toggleFileUploadVisibility(chatModeSelect.value);
            }
        }
    });
    
    // File upload event listener
    reportFileInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            uploadedFile = this.files[0];
            // Show file name in UI
            const fileUploadLabel = document.getElementById('file-upload-label');
            fileUploadLabel.innerHTML = `<i class="fas fa-file-pdf"></i> ${uploadedFile.name}`;
            fileUploadLabel.classList.add('file-selected');
        }
    });
    
    // Initialize file upload visibility based on current mode
    toggleFileUploadVisibility(chatModeSelect.value);
    
    // Function to send a message
    function sendMessage() {
        const messageText = userInput.value.trim();
        if ((!messageText && chatModeSelect.value !== 'report_analysis') || isTyping) return;
        
        // For report analysis mode, file is required
        if (chatModeSelect.value === 'report_analysis' && !uploadedFile) {
            alert('Please upload a PDF report file first.');
            return;
        }
        
        // Get the current chat mode
        const mode = chatModeSelect.value;
        
        // Add user message to the chat
        addMessageToChat('user', messageText || 'Analyze this health report please.', true, uploadedFile?.name);
        
        // Clear the input and reset its height
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Show the typing indicator
        showTypingIndicator();
        
        // Process the message based on the mode
        if (mode === 'health') {
            // For health-related queries
            sendHealthQuery(messageText);
        } else if (mode === 'general') {
            // For general queries
            sendGeneralQuery(messageText);
        } else if (mode === 'report_analysis') {
            // For health report analysis
            sendReportAnalysis(messageText || 'Analyze this health report please.', uploadedFile);
            
            // Reset file upload UI after sending
            uploadedFile = null;
            const fileUploadLabel = document.getElementById('file-upload-label');
            fileUploadLabel.innerHTML = '<i class="fas fa-file-upload"></i> Upload Report';
            fileUploadLabel.classList.remove('file-selected');
            reportFileInput.value = '';
        }
    }
    
    // Function to add a message to the chat interface
    function addMessageToChat(role, text, addToHistory = true, filename = null) {
        const messageContainer = document.createElement('div');
        messageContainer.className = `message ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        // Handle Markdown formatting for bot messages
        if (role === 'bot' && window.marked) {
            messageText.innerHTML = marked.parse(text);
        } else {
            // For user messages, keep as text
            const paragraph = document.createElement('p');
            paragraph.textContent = text;
            messageText.appendChild(paragraph);
        }
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = formatTime(new Date());
        
        messageContent.appendChild(messageText);
        messageContent.appendChild(messageTime);
        messageContainer.appendChild(messageContent);
        
        chatMessages.appendChild(messageContainer);
        
        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Add to chat history if needed
        if (addToHistory && currentChatId) {
            const chatIndex = chatHistory.findIndex(chat => chat.id === currentChatId);
            if (chatIndex !== -1) {
                chatHistory[chatIndex].messages.push({
                    role: role,
                    content: text,
                    timestamp: new Date().toISOString()
                });
                
                // Update chat title if it's the first user message
                if (role === 'user' && chatHistory[chatIndex].messages.filter(m => m.role === 'user').length === 1) {
                    // Set the chat title to the first 30 characters of the first user message
                    chatHistory[chatIndex].title = text.length > 30 ? text.substring(0, 30) + '...' : text;
                    renderChatHistory();
                }
                
                // Save to local storage
                saveChatsToLocalStorage();
            }
        }
        
        if (filename && role === 'user') {
            const fileAttachment = document.createElement('div');
            fileAttachment.className = 'file-attachment';
            fileAttachment.innerHTML = `<i class="fas fa-file-pdf"></i> ${filename}`;
            messageContent.appendChild(fileAttachment);
        }
    }
    
    // Function to show the typing indicator
    function showTypingIndicator() {
        isTyping = true;
        
        const typingContainer = document.createElement('div');
        typingContainer.className = 'message bot typing';
        typingContainer.id = 'typing-indicator';
        
        const typingContent = document.createElement('div');
        typingContent.className = 'message-content';
        
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingIndicator.appendChild(dot);
        }
        
        typingContent.appendChild(typingIndicator);
        typingContainer.appendChild(typingContent);
        
        chatMessages.appendChild(typingContainer);
        
        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to hide the typing indicator
    function hideTypingIndicator() {
        isTyping = false;
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    // Function to handle health-related queries
    function sendHealthQuery(query) {
        // Add health tag to the message
        const messageWithTag = "#health " + query;
        
        // Send the health query to the backend
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: messageWithTag, chat_id: currentChatId }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide the typing indicator
            hideTypingIndicator();
            
            // Display the response with any markdown formatting
            addMessageToChat('bot', data.response);
        })
        .catch(error => {
            hideTypingIndicator();
            console.error('Error:', error);
            addMessageToChat('bot', "I'm sorry, but I encountered an error processing your request. Please try again later.");
        });
    }
    
    // Function to handle general queries
    function sendGeneralQuery(query) {
        // Send the general query to the backend
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: query, chat_id: currentChatId }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide the typing indicator
            hideTypingIndicator();
            
            // Display the response
            addMessageToChat('bot', data.response);
        })
        .catch(error => {
            hideTypingIndicator();
            console.error('Error:', error);
            addMessageToChat('bot', "I'm sorry, but I encountered an error processing your request. Please try again later.");
        });
    }
    
    // Function to create a new chat
    function createNewChat() {
        // Generate a new unique ID
        const newChatId = 'chat_' + Date.now();
        
        // Add to chat history
        chatHistory.unshift({
            id: newChatId,
            title: 'New Chat',
            mode: chatModeSelect.value,
            created_at: new Date().toISOString(),
            messages: []
        });
        
        // Set as current chat
        currentChatId = newChatId;
        
        // Clear the chat interface
        chatMessages.innerHTML = '';
        
        // Add the welcome message based on the mode
        const welcomeMessage = chatModeSelect.value === 'health' 
            ? "Hello! I'm your health assistant. Please describe your symptoms or health concerns, and I'll try to help you." 
            : "Hello! I'm your AI assistant. How can I help you today?";
        
        addMessageToChat('bot', welcomeMessage, true);
        
        // Update the chat history sidebar
        renderChatHistory();
        
        // Save to local storage
        saveChatsToLocalStorage();
    }
    
    // Function to load a specific chat
    function loadChat(chatId) {
        // Find the chat in history
        const chat = chatHistory.find(chat => chat.id === chatId);
        if (!chat) return;
        
        // Set as current chat
        currentChatId = chatId;
        
        // Clear the chat interface
        chatMessages.innerHTML = '';
        
        // Set the chat mode
        chatModeSelect.value = chat.mode || 'health';
        
        // Render all messages
        for (const message of chat.messages) {
            if (message.role === 'bot' && typeof message.content === 'string' && message.content.startsWith('{')) {
                try {
                    // Handle special message types like diagnosis
                    const parsedContent = JSON.parse(message.content);
                    if (parsedContent.type === 'diagnosis') {
                        // For backward compatibility
                        const responseText = `Based on your symptoms, I think you might have: **${parsedContent.disease}** (confidence: ${parsedContent.confidence}%).
                        
**Important:** This is not a medical diagnosis. Always consult with a healthcare professional for proper diagnosis and treatment.`;
                        addMessageToChat('bot', responseText, false);
                    } else {
                        addMessageToChat(message.role, message.content, false);
                    }
                } catch (e) {
                    // If it's not valid JSON, treat as normal message
                    addMessageToChat(message.role, message.content, false);
                }
            } else {
                // Normal message
                addMessageToChat(message.role, message.content, false);
            }
        }
        
        // Update the chat history sidebar
        renderChatHistory();
    }
    
    // Function to render the chat history in the sidebar
    function renderChatHistory() {
        chatHistoryContainer.innerHTML = '';
        
        if (chatHistory.length === 0) {
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-history';
            emptyState.textContent = 'No chat history yet';
            chatHistoryContainer.appendChild(emptyState);
            return;
        }
        
        for (const chat of chatHistory) {
            const chatItem = document.createElement('div');
            chatItem.className = `chat-item ${chat.id === currentChatId ? 'active' : ''}`;
            chatItem.dataset.chatId = chat.id;
            
            const chatIcon = document.createElement('i');
            chatIcon.className = chat.mode === 'health' ? 'fas fa-heartbeat' : 'fas fa-comments';
            
            const chatTitle = document.createElement('span');
            chatTitle.textContent = chat.title || 'New Chat';
            
            chatItem.appendChild(chatIcon);
            chatItem.appendChild(chatTitle);
            
            chatItem.addEventListener('click', function() {
                loadChat(chat.id);
            });
            
            chatHistoryContainer.appendChild(chatItem);
        }
    }
    
    // Function to save chats to local storage
    function saveChatsToLocalStorage() {
        // Limit to the most recent 50 chats
        const chatsToSave = chatHistory.slice(0, 50);
        localStorage.setItem('chatHistory', JSON.stringify(chatsToSave));
    }
    
    // Function to load chats from local storage
    function loadChatHistory() {
        const savedChats = localStorage.getItem('chatHistory');
        if (savedChats) {
            chatHistory = JSON.parse(savedChats);
            if (chatHistory.length > 0) {
                currentChatId = chatHistory[0].id;
                loadChat(currentChatId);
            }
        }
    }
    
    // Helper function to format time
    function formatTime(date) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // Function to load the marked library for Markdown
    function loadMarkdownLibrary() {
        if (window.marked) return; // Already loaded
        
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
        script.onload = function() {
            // Configure marked options if needed
            if (window.marked) {
                marked.setOptions({
                    breaks: true, // Enable line breaks
                    gfm: true,    // Enable GitHub Flavored Markdown
                });
            }
        };
        document.head.appendChild(script);
    }
    
    function toggleFileUploadVisibility(mode) {
        if (mode === 'report_analysis') {
            fileUploadContainer.style.display = 'inline-block';
            userInput.placeholder = "Ask about the report or leave empty for general analysis...";
        } else {
            fileUploadContainer.style.display = 'none';
            userInput.placeholder = "Type your symptoms or questions here...";
        }
    }
    
    function sendReportAnalysis(query, file) {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('query', query);
        if (currentChatId) {
            formData.append('chat_id', currentChatId);
        }
        
        // Send request to the backend
        fetch('/api/upload-report', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide the typing indicator
            hideTypingIndicator();
            
            // Add response to the chat
            let responseText = data.response;
            
            // If there are sources, add them to the message
            if (data.sources && data.sources.length > 0) {
                responseText += '\n\n**Relevant Sections:**\n\n';
                data.sources.forEach((source, index) => {
                    // Truncate source if too long
                    const maxLength = 200;
                    const sourceTruncated = source.length > maxLength 
                        ? source.substring(0, maxLength) + '...' 
                        : source;
                    responseText += `${index + 1}. ${sourceTruncated}\n\n`;
                });
            }
            
            addMessageToChat('bot', responseText);
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            addMessageToChat('bot', 'Sorry, there was an error processing your report. Please try again or upload a different file.');
        });
    }
});