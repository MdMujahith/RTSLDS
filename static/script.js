document.addEventListener('DOMContentLoaded', function() {
    const socket = io();

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
    const aslIslBtn = document.getElementById('aslIslBtn');
    const aslBtn = document.getElementById('aslBtn');
    const islBtn = document.getElementById('islBtn');

    let isRunning = false;
    let detectionHistory = [];

    socket.on('connect', function() {
        console.log('Connected to server');
    });

    socket.on('video_frame', function(data) {
        if (isRunning) {
            videoElement.src = `data:image/jpeg;base64,${data.frame}`;
        }
    });

    socket.on('detection_update', function(data) {
        if (data.raw_detection) {
            detectedSigns.textContent = data.raw_detection;
        } else {
            detectedSigns.textContent = "No signs detected";
        }

        const confidencePercent = Math.round(data.confidence * 100);
        confidenceFill.style.width = `${confidencePercent}%`;
        confidenceValue.textContent = `${confidencePercent}%`;

        if (data.processed_text) {
            translatedText.textContent = data.processed_text;

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
        const audioSrc = `data:audio/mp3;base64,${data.audio_data}`;
        audioPlayer.src = audioSrc;
        audioPlayer.play().catch(e => console.error('Audio playback error:', e));
    });

    startStopBtn.addEventListener('click', function() {
        if (isRunning) {
            stopDetection();
        } else {
            startDetection();
        }
    });

    resetBtn.addEventListener('click', function() {
        socket.emit('reset_detection');
        detectedSigns.textContent = "Waiting for detection...";
        translatedText.textContent = "Waiting for translation...";
        confidenceFill.style.width = "0%";
        confidenceValue.textContent = "0%";
    });

    aslIslBtn.addEventListener('click', function() {
        setActiveButton(aslIslBtn);
        socket.emit('set_mode', { mode: 'both' });
    });

    aslBtn.addEventListener('click', function() {
        setActiveButton(aslBtn);
        socket.emit('set_mode', { mode: 'asl' });
    });

    islBtn.addEventListener('click', function() {
        setActiveButton(islBtn);
        socket.emit('set_mode', { mode: 'isl' });
    });

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
        [aslIslBtn, aslBtn, islBtn].forEach(btn => {
            btn.classList.remove('active');
        });
        activeBtn.classList.add('active');
    }

    function updateHistory() {
        historyContainer.innerHTML = '';
        if (detectionHistory.length === 0) {
            historyContainer.innerHTML = '<div class="text-muted text-center py-3">No history yet</div>';
            return;
        }

        detectionHistory.forEach(item => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.textContent = item;
            historyContainer.appendChild(historyItem);
        });
    }
});