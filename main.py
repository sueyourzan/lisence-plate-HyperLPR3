from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import hyperlpr3
import cv2
import numpy as np
import time
import io

app = FastAPI(title="智能车牌识别系统", version="1.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板引擎
templates = Jinja2Templates(directory="static")

# 初始化车牌识别器
catcher = hyperlpr3.LicensePlateCatcher()

# 新增：处理favicon.ico请求，避免404
@app.get("/favicon.ico")
async def favicon():
    # 返回空的204响应，或自定义图标
    return Response(status_code=204)

# 主页路由
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request})

# 车牌识别接口
@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({
                "success": False,
                "error": "无效的图片文件！"
            })

        start = time.time()
        results = catcher(img)
        cost_time = round(time.time() - start, 2)

        plate_results = []
        for code, conf, _, _ in results:
            plate_results.append({
                "number": str(code),
                "confidence": round(float(conf), 2)
            })

        return JSONResponse({
            "success": True,
            "time": cost_time,
            "results": plate_results
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)