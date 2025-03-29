document.addEventListener('DOMContentLoaded', function() {
    // Connect to Socket.IO server
    const socket = io();
    
    // Elements
    const videoElement = document.getElementById('videoElement');
    const startStopBtn = document.getElementById('startStopBtn');
    const startStopText = document.getElementById('startStopText');
    const captureBtn = document.getElementById('captureBtn');
    const resetBtn = document.getElementById('resetBtn');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const detectedSigns = document.getElementById('detectedSigns');
    const translatedText = document.getElementById('translatedText');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const historyContainer = document.getElementById('historyContainer');
    const audioPlayer = document.getElementById('audioPlayer');
    const aslBtn = document.getElementById('aslBtn');
    const islBtn = document.getElementById('islBtn');
    
    // State
    let isRunning = false;
    let detectionHistory = [];
    
    // Socket.IO events
    socket.on('connect', function() {
        console.log('Connected to server');
    });
    
    socket.on('status', function(data) {
        console.log('Status:', data.message);
    });
    
    socket.on('video_frame', function(data) {
        if (isRunning) {
            videoElement.src = `data:image/jpeg;base64,${data.frame}`;
        }
    });
    
    socket.on('detection_update', function(data) {
        // Update detected signs
        if (data.raw_detection) {
            detectedSigns.textContent = data.raw_detection;
        } else {
            detectedSigns.textContent = "No signs detected";
        }
        
        // Update confidence meter
        const confidencePercent = Math.round(data.confidence * 100);
        confidenceFill.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;
        
        // Update translated text
        if (data.processed_text) {
            translatedText.textContent = data.processed_text;
            
            // Add to history if it's new and not empty
            if (data.processed_text.trim() !== '' && !detectionHistory.includes(data.processed_text)) {
                detectionHistory.unshift(data.processed_text);
                if (detectionHistory.length > 5) {
                    detectionHistory.pop();
                }
                updateHistory();
            }
        } else {
            translatedText.textContent = "Waiting for translation...";
        }
    });
    
    socket.on('audio_update', function(data) {
        // Play the audio automatically
        const audioSrc = `data:audio/mp3;base64,${data.audio_data}`;
        audioPlayer.src = audioSrc;
        audioPlayer.play().catch(e => console.error('Audio playback error:', e));
    });
    
    // Button event listeners
    startStopBtn.addEventListener('click', function() {
        if (isRunning) {
            stopDetection();
        } else {
            startDetection();
        }
    });
    
    captureBtn.addEventListener('click', function() {
        // Implement capture functionality if needed
        console.log('Capture button clicked');
    });
    
    resetBtn.addEventListener('click', function() {
        socket.emit('reset_detection');
        detectedSigns.textContent = "Waiting for detection...";
        translatedText.textContent = "Waiting for translation...";
        confidenceFill.style.width = "0%";
        confidenceValue.textContent = "0%";
    });
    
    // Mode selection
    aslBtn.addEventListener('click', function() {
        setActiveButton(aslBtn);
        socket.emit('set_mode', { mode: 'asl' });
    });
    
    islBtn.addEventListener('click', function() {
        setActiveButton(islBtn);
        socket.emit('set_mode', { mode: 'isl' });
    });
    
    // Functions
    function startDetection() {
        socket.emit('start_detection');
        isRunning = true;
        startStopText.textContent = "Stop Detection";
        statusIndicator.classList.remove('status-inactive');
        statusIndicator.classList.add('status-active');
        statusText.textContent = "Detection Active";
    }
    
    function stopDetection() {
        socket.emit('stop_detection');
        isRunning = false;
        startStopText.textContent = "Start Detection";
        statusIndicator.classList.remove('status-active');
        statusIndicator.classList.add('status-inactive');
        statusText.textContent = "Detection Inactive";
    }
    
    function setActiveButton(activeBtn) {
        [aslBtn, islBtn].forEach(btn => {
            btn.classList.remove('active');
        });
        activeBtn.classList.add('active');
    }
    
    function updateHistory() {
        // Clear history container
        historyContainer.innerHTML = '';
        
        if (detectionHistory.length === 0) {
            historyContainer.innerHTML = '<div class="text-muted text-center py-3">No history yet</div>';
            return;
        }
        
        // Add each history item
        detectionHistory.forEach((item, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.textContent = item;
            historyContainer.appendChild(historyItem);
        });
    }
});