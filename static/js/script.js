document.addEventListener('DOMContentLoaded', function() {
    // --- Configuration ---
    const socket = io();
    
    // --- Elements ---
    const videoElement = document.getElementById('videoElement');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const startStopBtn = document.getElementById('startStopBtn');
    const startStopText = document.getElementById('startStopText');
    const captureBtn = document.getElementById('captureBtn');
    const resetBtn = document.getElementById('resetBtn');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const connectionStatus = document.getElementById('connectionStatus');
    
    const detectedSigns = document.getElementById('detectedSigns');
    const translatedText = document.getElementById('translatedText');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const historyContainer = document.getElementById('historyContainer');
    const audioPlayer = document.getElementById('audioPlayer');
    
    const aslBtn = document.getElementById('aslBtn');
    const islBtn = document.getElementById('islBtn');
    
    // --- State ---
    let isRunning = false;
    let detectionHistory = [];

    // --- Socket Events ---

    socket.on('connect', () => {
        console.log('✅ Connected to backend');
        connectionStatus.textContent = "Online";
        connectionStatus.className = "badge bg-success";
    });

    socket.on('disconnect', () => {
        console.log('❌ Disconnected');
        connectionStatus.textContent = "Offline";
        connectionStatus.className = "badge bg-danger";
        stopDetectionUI(); // Safety: stop UI if connection drops
    });

    // Receive Video Stream
    socket.on('video_frame', (data) => {
        if (isRunning) {
            // Switch placeholder -> video
            videoPlaceholder.style.display = 'none';
            videoElement.style.display = 'block';
            videoElement.src = `data:image/jpeg;base64,${data.frame}`;
        }
    });

    // Receive Detection Data
    socket.on('detection_update', (data) => {
        // 1. Update Sign Text
        detectedSigns.textContent = data.raw_detection || "...";
        
        // 2. Update Confidence
        const confidence = Math.round((data.confidence || 0) * 100);
        confidenceFill.style.width = `${confidence}%`;
        confidenceValue.textContent = `${confidence}%`;
        
        // Color code the confidence bar
        if (confidence > 80) confidenceFill.className = "progress-bar bg-success";
        else if (confidence > 50) confidenceFill.className = "progress-bar bg-warning";
        else confidenceFill.className = "progress-bar bg-danger";

        // 3. Update Sentence & History
        if (data.processed_text) {
            translatedText.textContent = data.processed_text;
            
            // Add to history if it's a new unique sentence
            if (data.processed_text.trim() !== '' && !detectionHistory.includes(data.processed_text)) {
                addToHistory(data.processed_text);
            }
        }
    });

    // Receive Audio
    socket.on('audio_update', (data) => {
        try {
            audioPlayer.src = `data:audio/mp3;base64,${data.audio_data}`;
            audioPlayer.play().catch(e => console.warn("Auto-play blocked:", e));
        } catch (err) {
            console.error("Audio error:", err);
        }
    });

    // --- Button Actions ---

    startStopBtn.addEventListener('click', () => {
        if (isRunning) stopDetection();
        else startDetection();
    });

    resetBtn.addEventListener('click', () => {
        socket.emit('reset_detection');
        detectedSigns.textContent = "...";
        translatedText.textContent = "Waiting for input...";
        confidenceFill.style.width = "0%";
        confidenceValue.textContent = "0%";
        clearHistoryUI();
    });

    // Capture Screenshot functionality
    captureBtn.addEventListener('click', () => {
        if (!isRunning || !videoElement.src) return;
        
        const link = document.createElement('a');
        link.download = `sign_language_${new Date().getTime()}.jpg`;
        link.href = videoElement.src;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    aslBtn.addEventListener('click', () => {
        setActiveButton(aslBtn);
        socket.emit('set_mode', { mode: 'asl' });
    });

    islBtn.addEventListener('click', () => {
        setActiveButton(islBtn);
        socket.emit('set_mode', { mode: 'isl' });
    });

    // --- Helper Functions ---

    function startDetection() {
        socket.emit('start_detection');
        isRunning = true;
        
        // UI Updates
        startStopBtn.innerHTML = '<i class="bi bi-stop-fill"></i> Stop Detection';
        startStopBtn.classList.replace('btn-primary', 'btn-danger');
        
        captureBtn.disabled = false;
        
        statusIndicator.classList.replace('status-inactive', 'status-active');
        statusText.textContent = "Detecting...";
    }

    function stopDetection() {
        socket.emit('stop_detection');
        stopDetectionUI();
    }

    function stopDetectionUI() {
        isRunning = false;
        
        // Switch video -> placeholder
        videoElement.style.display = 'none';
        videoPlaceholder.style.display = 'block';
        videoElement.src = ""; 

        // UI Updates
        startStopBtn.innerHTML = '<i class="bi bi-play-fill"></i> Start Detection';
        startStopBtn.classList.replace('btn-danger', 'btn-primary');
        
        captureBtn.disabled = true;
        
        statusIndicator.classList.replace('status-active', 'status-inactive');
        statusText.textContent = "Ready";
    }

    function setActiveButton(activeBtn) {
        [aslBtn, islBtn].forEach(btn => btn.classList.remove('active'));
        activeBtn.classList.add('active');
    }

    function addToHistory(text) {
        // Remove empty state message
        if (detectionHistory.length === 0) historyContainer.innerHTML = '';

        detectionHistory.unshift(text);
        if (detectionHistory.length > 20) detectionHistory.pop(); // Max 20 items

        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `<strong>${text}</strong> <small class="text-muted d-block" style="font-size:0.75rem">${new Date().toLocaleTimeString()}</small>`;
        
        historyContainer.prepend(div);
    }

    function clearHistoryUI() {
        detectionHistory = [];
        historyContainer.innerHTML = '<div class="text-center text-muted py-5 small"><i class="bi bi-clock-history mb-2 d-block fs-4"></i>History cleared</div>';
    }
});