# 🚗 智能车牌识别系统

基于 [HyperLPR3](https://github.com/szad670401/HyperLPR) 的深度学习中文车牌识别系统，提供 FastAPI 后端 API 和多种风格的 Web 前端界面。

## ✨ 功能特性

- **AI 深度学习识别** — 基于 ONNX Runtime 的高精度车牌检测与识别
- **多颜色车牌支持** — 蓝牌、黄牌、绿牌、白牌、黑牌、新能源绿牌
- **前后端分离架构** — FastAPI RESTful API + 纯前端页面
- **实时识别耗时统计** — 毫秒级响应
- **多种 UI 主题** — 浅色简约风 / 暗黑科技风 / 玻璃拟态风
- **历史记录** — 浏览器 localStorage 保存识别历史

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习引擎 | [HyperLPR3](https://github.com/szad670401/HyperLPR) |
| 推理框架 | ONNX Runtime |
| 图像处理 | OpenCV |
| Web 框架 | FastAPI + Uvicorn |
| 前端 | 原生 HTML/CSS/JS（Bootstrap / Font Awesome） |

## 📁 项目结构

```
lisence_plate/
├── app.py                  # 🔵 推荐主入口：完整 API 版（CORS + 健康检查 + 颜色识别）
├── main.py                 # Jinja2 模板版入口（使用 static/index1.html）
├── license_plate.py        # 极简版：内嵌 HTML，单文件运行
├── requirements.txt        # Python 依赖
├── static/
│   ├── index.html          # 前后端分离版 UI（暗黑科技风，调用 /api/detect_plate）
│   └── index1.html         # 模板版 UI（玻璃拟态风，调用 /detect）
└── README.md
```

> **推荐使用 `app.py` 作为主入口**，它具备最完整的 API 设计（CORS 跨域、健康检查、车牌颜色映射、完善的错误处理）。

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- pip

### 2. 安装依赖

```bash
# 安装 HyperLPR3（核心识别引擎）
pip install hyperlpr3

# 安装其他依赖
pip install -r requirements.txt
```

> ⚠️ `requirements.txt` 中未包含 `hyperlpr3`，请务必先手动安装。

### 3. 启动服务

```bash
# 推荐方式：完整 API 版
python app.py

# 或极简版（内嵌 HTML）
python license_plate.py
```

服务启动后访问：**http://127.0.0.1:8000**

### 4. 使用

1. 浏览器打开 `http://127.0.0.1:8000`
2. 选择/拖拽一张包含车牌的图片
3. 点击「开始识别」
4. 查看识别结果（车牌号、置信度、颜色、耗时）

## 📡 API 接口

### 健康检查

```
GET /api/health
```

响应：
```json
{
    "status": "success",
    "message": "车牌识别API服务运行正常",
    "timestamp": 1703664000.0
}
```

### 车牌识别

```
POST /api/detect_plate
Content-Type: multipart/form-data
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `file` | File | 车牌图片（JPG/PNG/BMP） |

响应：
```json
{
    "success": true,
    "detect_time": 0.15,
    "results": [
        {
            "plate_number": "京A12345",
            "confidence": 0.98,
            "color": "蓝牌",
            "box": [100, 200, 300, 400]
        }
    ],
    "count": 1
}
```

## 🎨 UI 预览

系统内置 3 种不同风格的 Web UI：

| 文件 | 风格 | 入口 |
|------|------|------|
| `app.py` 根路径 | 暗黑科技风（Bootstrap） | `http://127.0.0.1:8000/` |
| `license_plate.py` | 浅色简约风 | `http://127.0.0.1:8000/` |
| `main.py` + `index1.html` | 玻璃拟态霓虹风 | `http://127.0.0.1:8000/` |

## 📝 注意事项

1. **HyperLPR3 需单独安装**：`pip install hyperlpr3`（不在 requirements.txt 中）
2. **首次运行会下载模型**：HyperLPR3 首次启动会自动下载 ONNX 模型文件
3. **图片格式**：支持 JPG、PNG、BMP 格式
4. **生产部署**：建议将 `allow_origins=["*"]` 改为具体的前端域名
5. **Python 版本**：`__pycache__` 中包含 Python 3.9 和 3.10 缓存，建议统一 Python 版本

## 📄 License

仅供学习和研究使用。
