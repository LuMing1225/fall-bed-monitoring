/**
 * 跌倒/在床行为精准判别系统 - 前端逻辑
 */

// ============ 配置 ============
const BACKEND_URL = `http://${window.location.hostname}:8564`;
const WS_URL = `ws://${window.location.hostname}:8564/ws/detect`;

// ============ DOM 元素 ============
const elements = {
    video: document.getElementById('videoPlayer'),
    canvas: document.getElementById('overlayCanvas'),
    placeholder: document.getElementById('videoPlaceholder'),
    videoInput: document.getElementById('videoInput'),
    cameraBtn: document.getElementById('cameraBtn'),
    playBtn: document.getElementById('playBtn'),
    stopBtn: document.getElementById('stopBtn'),
    resetBedBtn: document.getElementById('resetBedBtn'),
    resetBtn: document.getElementById('resetBtn'),
    connectionStatus: document.getElementById('connectionStatus'),
    modelStatus: document.getElementById('modelStatus'),
    fallAlarmOverlay: document.getElementById('fallAlarmOverlay'),
    fallAlarmTime: document.getElementById('fallAlarmTime'),
    progressContainer: document.getElementById('progressContainer'),
    progressBar: document.getElementById('progressBar'),
    progressText: document.getElementById('progressText'),
    fpsValue: document.getElementById('fpsValue'),
    latencyValue: document.getElementById('latencyValue'),
    personCount: document.getElementById('personCount'),
    fallCount: document.getElementById('fallCount'),
    fallMetricCard: document.getElementById('fallMetricCard'),
    activeBar: document.getElementById('activeBar'),
    onbedBar: document.getElementById('onbedBar'),
    fallenBar: document.getElementById('fallenBar'),
    activeCount: document.getElementById('activeCount'),
    onbedCount: document.getElementById('onbedCount'),
    fallenCount: document.getElementById('fallenCount'),
    personList: document.getElementById('personList'),
    samStatus: document.getElementById('samStatus'),
    poseStatus: document.getElementById('poseStatus'),
    llmStatus: document.getElementById('llmStatus'),
    eventLog: document.getElementById('eventLog'),
    currentTime: document.getElementById('currentTime'),
};

const ctx = elements.canvas.getContext('2d');

// ============ 状态 ============
let ws = null;
let isPlaying = false;
let isCamera = false;
let mediaStream = null;
let animationId = null;
let frameCount = 0;
let totalFallEvents = 0;
let stateStats = { active: 0, on_bed: 0, fallen: 0, unknown: 0 };
let lastFrameData = null;

// ============ 工具函数 ============
function addLog(message, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const time = new Date().toLocaleTimeString('zh-CN');
    entry.textContent = `[${time}] ${message}`;
    elements.eventLog.insertBefore(entry, elements.eventLog.firstChild);
    // 最多保留50条
    while (elements.eventLog.children.length > 50) {
        elements.eventLog.removeChild(elements.eventLog.lastChild);
    }
}

function updateClock() {
    elements.currentTime.textContent = new Date().toLocaleString('zh-CN');
}
setInterval(updateClock, 1000);
updateClock();

// ============ WebSocket 连接 ============
function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    try {
        ws = new WebSocket(WS_URL);
    } catch (e) {
        addLog(`WebSocket连接失败: ${e.message}`, 'error');
        setTimeout(connectWebSocket, 3000);
        return;
    }

    ws.onopen = () => {
        elements.connectionStatus.className = 'status-badge connected';
        elements.connectionStatus.querySelector('.status-text').textContent = '已连接';
        addLog('WebSocket连接建立', 'success');
        // 发送ping
        ws.send(JSON.stringify({ type: 'ping' }));
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('WebSocket消息解析失败:', e);
        }
    };

    ws.onclose = () => {
        elements.connectionStatus.className = 'status-badge disconnected';
        elements.connectionStatus.querySelector('.status-text').textContent = '已断开';
        addLog('WebSocket连接断开，3秒后重连...', 'warning');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket错误:', error);
    };
}

function handleWebSocketMessage(data) {
    if (data.type === 'result') {
        lastFrameData = data;
        updateUI(data.result);
        // 显示标注图
        if (data.annotated_image) {
            displayAnnotatedImage(data.annotated_image);
        }
    } else if (data.type === 'pong') {
        // 心跳响应
    } else if (data.type === 'status') {
        addLog(data.message, 'info');
    } else if (data.error) {
        addLog(`服务端错误: ${data.error}`, 'error');
    }
}

function displayAnnotatedImage(dataUrl) {
    const img = new Image();
    img.onload = () => {
        elements.canvas.width = img.width;
        elements.canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };
    img.src = dataUrl;
}

// ============ UI 更新 ============
function updateUI(result) {
    // 指标
    elements.fpsValue.textContent = result.fps ? result.fps.toFixed(1) : '--';
    elements.latencyValue.textContent = result.processing_time_ms ? result.processing_time_ms.toFixed(1) : '--';
    elements.personCount.textContent = result.person_count || 0;

    // 跌倒检测
    if (result.fall_detected) {
        totalFallEvents++;
        elements.fallCount.textContent = totalFallEvents;
        elements.fallMetricCard.classList.add('alarm');
        elements.fallAlarmOverlay.classList.add('active');
        elements.fallAlarmTime.textContent = `第 ${result.frame_idx} 帧 | ${result.timestamp.toFixed(1)}s`;
        addLog(`检测到跌倒事件! 帧:${result.frame_idx}`, 'error');
        setTimeout(() => {
            elements.fallAlarmOverlay.classList.remove('active');
        }, 3000);
    } else {
        elements.fallMetricCard.classList.remove('alarm');
    }

    // 状态统计
    if (result.persons) {
        for (const p of result.persons) {
            const state = p.classification.state;
            stateStats[state] = (stateStats[state] || 0) + 1;
        }
    }
    updateStateBars();

    // 人员详情
    updatePersonList(result.persons || []);

    // 进度
    if (isPlaying && !isCamera) {
        frameCount = result.frame_idx;
        updateProgress();
    }
}

function updateStateBars() {
    const total = stateStats.active + stateStats.on_bed + stateStats.fallen + stateStats.unknown;
    if (total === 0) return;

    elements.activeCount.textContent = stateStats.active;
    elements.onbedCount.textContent = stateStats.on_bed;
    elements.fallenCount.textContent = stateStats.fallen;

    elements.activeBar.style.width = `${(stateStats.active / total) * 100}%`;
    elements.onbedBar.style.width = `${(stateStats.on_bed / total) * 100}%`;
    elements.fallenBar.style.width = `${(stateStats.fallen / total) * 100}%`;
}

function updatePersonList(persons) {
    if (persons.length === 0) {
        elements.personList.innerHTML = '<div class="empty-state">暂无检测数据</div>';
        return;
    }

    elements.personList.innerHTML = persons.map(p => {
        const cls = p.classification;
        const details = cls.details || {};
        const stateClass = cls.state;
        const stateLabel = cls.label;
        return `
            <div class="person-item ${stateClass}">
                <div class="person-header">
                    <span class="person-id">人员 #${p.track_id}</span>
                    <span class="person-state ${stateClass}">${stateLabel} ${(cls.confidence * 100).toFixed(0)}%</span>
                </div>
                <div class="person-metrics">
                    <span>IoU <strong>${(details.iou || 0).toFixed(3)}</strong></span>
                    <span>在床点比 <strong>${((details.keypoint_in_bed_ratio || 0) * 100).toFixed(0)}%</strong></span>
                    <span>轴线角 <strong>${details.body_axis_angle >= 0 ? details.body_axis_angle.toFixed(1) + '°' : '--'}</strong></span>
                    <span>臀部Y <strong>${details.hip_y ? details.hip_y.toFixed(0) : '--'}</strong></span>
                </div>
            </div>
        `;
    }).join('');
}

function updateProgress() {
    const video = elements.video;
    if (video.duration) {
        const percent = (video.currentTime / video.duration) * 100;
        elements.progressBar.style.width = `${percent}%`;
        elements.progressText.textContent = `${video.currentTime.toFixed(1)}s / ${video.duration.toFixed(1)}s`;
    }
}

// ============ 视频帧发送 ============
function sendFrameToBackend() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!isPlaying) return;

    const video = elements.video;
    if (video.readyState < 2 || video.paused || video.ended) return;

    // 绘制当前帧到临时canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    // 压缩为JPEG
    const jpegData = tempCanvas.toDataURL('image/jpeg', 0.8);

    // 发送
    ws.send(JSON.stringify({
        type: 'frame',
        image: jpegData,
    }));
}

// 视频帧捕获循环（控制发送频率约15fps）
let lastSendTime = 0;
function frameCaptureLoop() {
    if (!isPlaying) return;

    const now = Date.now();
    if (now - lastSendTime >= 66) { // ~15fps
        sendFrameToBackend();
        lastSendTime = now;
    }

    animationId = requestAnimationFrame(frameCaptureLoop);
}

// ============ 事件处理 ============

// 视频上传
elements.videoInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    stopPlayback();
    const url = URL.createObjectURL(file);
    elements.video.src = url;
    elements.video.style.display = 'block';
    elements.placeholder.style.display = 'none';
    elements.playBtn.disabled = false;
    elements.progressContainer.style.display = 'flex';
    isCamera = false;
    addLog(`视频已加载: ${file.name}`, 'success');
});

// 摄像头
elements.cameraBtn.addEventListener('click', async () => {
    if (isCamera) {
        stopCamera();
        return;
    }

    try {
        addLog('正在打开摄像头...', 'info');
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 },
            audio: false
        });
        elements.video.srcObject = mediaStream;
        elements.video.style.display = 'block';
        elements.placeholder.style.display = 'none';
        elements.video.play();
        isCamera = true;
        isPlaying = true;
        elements.cameraBtn.classList.add('active');
        elements.cameraBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
                <rect x="4" y="4" width="10" height="10" rx="1"/>
            </svg>
            关闭摄像头`;
        elements.stopBtn.disabled = false;
        elements.progressContainer.style.display = 'none';
        addLog('摄像头已开启', 'success');
        frameCaptureLoop();
    } catch (err) {
        addLog(`摄像头打开失败: ${err.message}`, 'error');
    }
});

// 播放
elements.playBtn.addEventListener('click', () => {
    if (elements.video.paused) {
        elements.video.play();
        isPlaying = true;
        elements.playBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
                <rect x="4" y="3" width="3" height="12" rx="1"/>
                <rect x="11" y="3" width="3" height="12" rx="1"/>
            </svg>
            暂停`;
        elements.stopBtn.disabled = false;
        addLog('视频播放开始', 'info');
        frameCaptureLoop();
    } else {
        elements.video.pause();
        isPlaying = false;
        elements.playBtn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
                <path d="M5 3L15 9L5 15V3Z"/>
            </svg>
            播放`;
        addLog('视频暂停', 'info');
    }
});

// 停止
elements.stopBtn.addEventListener('click', () => {
    stopPlayback();
    if (isCamera) stopCamera();
});

function stopPlayback() {
    isPlaying = false;
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    elements.video.pause();
    elements.video.currentTime = 0;
    elements.playBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
            <path d="M5 3L15 9L5 15V3Z"/>
        </svg>
        播放`;
    elements.stopBtn.disabled = true;
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
}

function stopCamera() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }
    elements.video.srcObject = null;
    elements.video.style.display = 'none';
    elements.placeholder.style.display = 'flex';
    isCamera = false;
    isPlaying = false;
    elements.cameraBtn.classList.remove('active');
    elements.cameraBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <rect x="2" y="5" width="10" height="8" rx="1.5" stroke="currentColor" stroke-width="2"/>
            <path d="M12 8L16 5.5V12.5L12 10" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
        </svg>
        摄像头`;
    addLog('摄像头已关闭', 'info');
}

// 重置床体
elements.resetBedBtn.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'reset_bed' }));
    }
    addLog('床体掩膜已重置，下一帧将重新分割', 'warning');
});

// 重置系统
elements.resetBtn.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'reset' }));
    }
    totalFallEvents = 0;
    stateStats = { active: 0, on_bed: 0, fallen: 0, unknown: 0 };
    elements.fallCount.textContent = '0';
    updateStateBars();
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    addLog('系统已重置', 'warning');
});

// 视频结束
elements.video.addEventListener('ended', () => {
    isPlaying = false;
    elements.playBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor">
            <path d="M5 3L15 9L5 15V3Z"/>
        </svg>
        播放`;
    addLog('视频播放结束', 'info');
});

// ============ 系统状态轮询 ============
async function checkSystemStatus() {
    try {
        const resp = await fetch(`${BACKEND_URL}/api/status`);
        const status = await resp.json();

        if (status.models_loaded) {
            elements.modelStatus.className = 'model-badge ready';
            elements.modelStatus.querySelector('span').textContent = '模型已就绪';
            elements.samStatus.textContent = '运行中';
            elements.samStatus.className = 'algo-status ok';
            elements.poseStatus.textContent = '运行中';
            elements.poseStatus.className = 'algo-status ok';
        } else {
            elements.modelStatus.querySelector('span').textContent = '模型加载中...';
        }

        if (status.llm_agent) {
            if (status.llm_agent.ollama_available) {
                elements.llmStatus.textContent = '已连接';
                elements.llmStatus.className = 'algo-status ok';
            } else {
                elements.llmStatus.textContent = '未连接(纯算法)';
                elements.llmStatus.className = 'algo-status warn';
            }
        }
    } catch (e) {
        elements.modelStatus.querySelector('span').textContent = '后端未响应';
    }
}

// ============ 初始化 ============
function init() {
    addLog('系统初始化中...', 'info');
    connectWebSocket();
    checkSystemStatus();
    setInterval(checkSystemStatus, 5000);
    addLog('前端初始化完成，请上传视频或开启摄像头', 'success');
}

init();
