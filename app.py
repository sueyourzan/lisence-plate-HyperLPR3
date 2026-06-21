from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # 解决跨域问题
import hyperlpr3
import cv2
import numpy as np
import time
import io
from hyperlpr3 import hyperLPR3

# 初始化FastAPI应用
app = FastAPI(title="车牌识别API服务", version="1.0")

# 解决跨域问题：允许前端页面访问API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源，生产环境需指定前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化车牌识别引擎
catcher = hyperlpr3.LicensePlateCatcher()
# 车牌类型映射
PLATE_TYPE_MAP = {
    0: "蓝牌", 1: "黄牌", 2: "绿牌", 3: "白牌", 4: "黑牌", 5: "新能源绿牌"
}


# 健康检查接口（用于测试服务是否正常）
@app.get("/api/health")
async def health_check():
    return {
        "status": "success",
        "message": "车牌识别API服务运行正常",
        "timestamp": time.time()
    }


# 车牌识别核心接口
@app.post("/api/detect_plate")
async def detect_plate(file: UploadFile = File(...)):
    try:
        # 1. 读取并解析上传的图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cv_img is None:
            return {
                "success": False,
                "error": "无法解析图片，请上传有效的JPG/PNG/BMP格式图片"
            }

        # 2. 执行车牌识别并统计耗时
        start_time = time.time()
        results = catcher(cv_img)
        detect_time = round(float(time.time() - start_time), 2)

        # 3. 处理识别结果（转换所有numpy类型为Python原生类型）
        plate_results = []
        for code, confidence, type_idx, box in results:
            plate_results.append({
                "plate_number": str(code),
                "confidence": round(float(confidence), 2),
                "color": PLATE_TYPE_MAP.get(int(type_idx), "未知"),
                "box": [int(x) for x in box]
            })

        # 4. 返回标准化JSON响应
        return {
            "success": True,
            "detect_time": detect_time,
            "results": plate_results,
            "count": len(plate_results)
        }

    except Exception as e:
        # 异常处理：返回友好的错误信息
        error_msg = str(e)
        if "hyperlpr3" in error_msg.lower() or "model" in error_msg.lower():
            error_msg = "车牌识别模型加载失败，请检查HyperLPR3安装"
        elif "cv2" in error_msg.lower():
            error_msg = "图片处理失败，仅支持JPG/PNG/BMP格式"

        return {
            "success": False,
            "detect_time": 0.0,
            "error": error_msg
        }


# 启动服务（核心修复：移除模块名中的.py后缀）
if __name__ == "__main__":
    import uvicorn


    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)