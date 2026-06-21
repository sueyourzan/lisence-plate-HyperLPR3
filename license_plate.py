# 极简版车牌识别系统（前后端不分离 + 无颜色功能 + 历史记录 + UI优化）
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import hyperlpr3
import cv2
import numpy as np
import time
import io

app = FastAPI()
catcher = hyperlpr3.LicensePlateCatcher()

# ===================== 优化后的前端页面 =====================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能车牌识别系统</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4edf9 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 25px;
            padding: 20px;
        }
        .header h1 {
            color: #1890ff;
            font-size: 2.4rem;
            margin-bottom: 8px;
        }
        .header p {
            color: #666;
            font-size: 1rem;
        }

        .main-container {
            display: flex;
            gap: 24px;
            max-width: 1400px;
            margin: 0 auto;
            flex-wrap: wrap;
        }

        .panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
            padding: 24px;
            transition: transform 0.2s ease;
        }
        .panel:hover {
            transform: translateY(-4px);
        }

        .left-panel {
            flex: 1;
            min-width: 320px;
        }
        .right-panel {
            flex: 2;
            min-width: 400px;
        }
        .history-panel {
            flex: 1;
            min-width: 320px;
            max-height: 600px;
            overflow-y: auto;
        }

        .btn {
             display: block;
             width: 100%;
             padding: 12px;
             border: none;
             border-radius: 8px;
             font-size: 16px;
             font-weight: 500;
             cursor: pointer;
             margin-bottom: 12px;
             transition: all 0.2s ease;
             text-align: center;
        }
        .btn-upload {
            background: #1890ff;
            color: white;
        }
        .btn-upload:hover:not(:disabled) {
            background: #40a9ff;
        }
        .btn-detect {
            background: #52c41a;
            color: white;
        }
        .btn-detect:hover:not(:disabled) {
            background: #73d13d;
        }
        .btn-clear {
            background: #ff7875;
            color: white;
        }
        .btn-clear:hover:not(:disabled) {
            background: #ff9c99;
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        #file-input {
            display: none;
        }

        .preview-container {
            text-align: center;
            margin-top: 12px;
        }
        .preview-img {
            max-width: 100%;
            max-height: 450px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            object-fit: contain;
            background: #fafafa;
        }

        .result-area {
            min-height: 120px;
            padding: 16px;
            border-radius: 8px;
            margin-top: 16px;
            background: #f9f9f9;
            border: 1px dashed #d9d9d9;
        }
        .result-item {
            background: #f0f9ff;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #1890ff;
        }
        .time-display {
            font-size: 14px;
            color: #8c8c8c;
            margin-top: 8px;
            text-align: center;
        }

        .loading {
            display: none;
            text-align: center;
            color: #1890ff;
            font-weight: 500;
            margin: 12px 0;
        }

        .history-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .history-list {
            list-style: none;
        }
        .history-item {
            padding: 12px;
            border-bottom: 1px solid #eee;
            font-size: 14px;
            animation: fadeIn 0.3s ease;
        }
        .history-item:last-child {
            border-bottom: none;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .history-number {
            font-weight: bold;
            color: #1890ff;
        }
        .history-time {
            color: #8c8c8c;
            font-size: 12px;
            margin-top: 4px;
        }

        @media (max-width: 900px) {
            .main-container {
                flex-direction: column;
            }
            .right-panel {
                order: -1;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><i class="fas fa-car"></i> 智能车牌识别系统</h1>
        <p>上传图片，快速识别车牌号码（本地运行 · 无网络请求）</p>
    </div>

    <div class="main-container">
        <!-- 左侧：操作区 -->
        <div class="left-panel">
            <label for="file-input" class="btn btn-upload">
                <i class="fas fa-folder-open"></i> 选择车牌图片
            </label>
            <button class="btn btn-detect" id="detect-btn">
                 <i class="fas fa-search"></i> 开始识别
            </button>

            <div class="loading" id="loading">识别中，请稍候...</div>
            <div class="time-display">识别耗时：<span id="detect-time">0.00</span> 秒</div>
            <div class="result-area" id="result-area">
                 请上传图片并点击识别按钮
            </div>
        </div>

        <!-- 右侧：图片预览 -->
        <div class="panel right-panel">
            <h3 style="text-align: center; margin-bottom: 16px; color: #333;">
                <i class="far fa-image"></i> 图片预览
            </h3>
            <div class="preview-container">
                <img id="preview-img" class="preview-img" src="" alt="预览图">
            </div>
        </div>

        <!-- 右侧：历史记录 -->
        <div class="panel history-panel">
            <div class="history-title">
                <h3><i class="fas fa-history"></i> 识别历史</h3>
                <button class="btn btn-clear" id="clear-history" style="padding: 6px 12px; font-size: 14px;">
                    <i class="fas fa-trash"></i> 清空
                </button>
            </div>
            <ul class="history-list" id="history-list">
                <li style="text-align: center; color: #999; padding: 20px;">暂无记录</li>
            </ul>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('file-input');
        const detectBtn = document.getElementById('detect-btn');
        const previewImg = document.getElementById('preview-img');
        const resultArea = document.getElementById('result-area');
        const loading = document.getElementById('loading');
        const detectTime = document.getElementById('detect-time');
        const historyList = document.getElementById('history-list');
        const clearHistoryBtn = document.getElementById('clear-history');

        // 加载历史记录
        function loadHistory() {
            const records = JSON.parse(localStorage.getItem('plateRecords') || '[]');
            renderHistory(records);
        }

        function renderHistory(records) {
            if (records.length === 0) {
                historyList.innerHTML = '<li style="text-align: center; color: #999; padding: 20px;">暂无记录</li>';
                return;
            }
            historyList.innerHTML = '';
            // 倒序显示（最新在上）
            records.slice().reverse().forEach(record => {
                const li = document.createElement('li');
                li.className = 'history-item';
                li.innerHTML = `
                    <div><span class="history-number">${record.number}</span> (置信度: ${record.confidence})</div>
                    <div class="history-time">⏱️ ${record.time}s | ${record.timestamp}</div>
                `;
                historyList.appendChild(li);
            });
        }

        // 保存记录到 localStorage
        function saveRecord(number, confidence, timeCost) {
            const records = JSON.parse(localStorage.getItem('plateRecords') || '[]');
            const newRecord = {
                number,
                confidence,
                time: timeCost,
                timestamp: new Date().toLocaleString('zh-CN')
            };
            records.push(newRecord);
            // 保留最近50条
            if (records.length > 50) records.shift();
            localStorage.setItem('plateRecords', JSON.stringify(records));
            loadHistory(); // 刷新显示
        }

        // 预览图片
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImg.src = e.target.result;
                    resultArea.innerHTML = "✅ 图片已加载，点击“开始识别”";
                    detectTime.textContent = "0.00";
                };
                reader.readAsDataURL(file);
            }
        });

        // 识别逻辑
        detectBtn.addEventListener('click', async () => {
            const file = fileInput.files[0];
            if (!file) {
                resultArea.innerHTML = "<span style='color: #ff4d4f;'>⚠️ 请先选择一张图片！</span>";
                return;
            }

            loading.style.display = "block";
            detectBtn.disabled = true;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                loading.style.display = "none";
                detectBtn.disabled = false;

                if (data.success) {
                    detectTime.textContent = data.time;
                    if (data.results.length === 0) {
                        resultArea.innerHTML = "<span style='color: #fa8c16;'>🔍 未检测到车牌，请尝试清晰图片</span>";
                    } else {
                        let html = `<h4 style="margin: 0 0 12px 0; color: #1890ff;">识别成功！</h4>`;
                        data.results.forEach((item, idx) => {
                            html += `
                                <div class="result-item">
                                    <strong>车牌 ${idx + 1}：</strong> <span class="history-number">${item.number}</span><br>
                                    <strong>置信度：</strong> ${item.confidence}
                                </div>
                            `;
                            // 保存每条记录
                            saveRecord(item.number, item.confidence, data.time);
                        });
                        resultArea.innerHTML = html;
                    }
                } else {
                    resultArea.innerHTML = `<span style='color: #ff4d4f;'>❌ 识别失败：${data.error}</span>`;
                }
            } catch (error) {
                loading.style.display = "none";
                detectBtn.disabled = false;
                resultArea.innerHTML = `<span style='color: #ff4d4f;'>💥 请求出错：${error.message}</span>`;
                console.error("识别错误:", error);
            }
        });

        // 清空历史
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('确定要清空所有识别记录吗？')) {
                localStorage.removeItem('plateRecords');
                loadHistory();
            }
        });

        // 初始化
        loadHistory();
    </script>
</body>
</html>
"""


# ===================== 后端接口（无需修改，但保持一致性） =====================
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "无效的图片文件！"}

        start = time.time()
        results = catcher(img)
        cost_time = round(time.time() - start, 2)

        plate_results = []
        for code, conf, _, _ in results:
            plate_results.append({
                "number": str(code),
                "confidence": round(float(conf), 2)
            })

        return {
            "success": True,
            "time": cost_time,
            "results": plate_results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("license_plate:app", host="127.0.0.1", port=8000, reload=False)