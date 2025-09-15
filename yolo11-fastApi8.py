from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import base64
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
import asyncio
import json
import time
import os
from fastapi.responses import HTMLResponse
# 自定义文档
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
# 初始化 FastAPI 应用
app = FastAPI(
    title="YOLO Object Detection API",
    description="基于 YOLO 的图像目标检测服务，支持多摄像头实时检测和图片上传检测",
    version="1.0.0",
    docs_url=None,  # 禁用默认的 docs
    redoc_url=None,  # 禁用默认的 redoc
    openapi_url="/openapi.json"
)
'''
 挂载静态文件服务，使应用能够提供Swagger UI所需的本地JavaScript和CSS文件。
 将URL路径 /static 映射到磁盘上的 static 目录
 为自定义文档界面提供必要的静态资源支持
 实现离线可用的API文档功能
'''
app.mount("/static", StaticFiles(directory="static"), name="static")


# 跨域资源共享(CORS)配置，用于解决前端跨域访问问题。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头部
    expose_headers=["*"]  # 暴露所有响应头部
)

# 配置FastAPI应用的最大请求体大小限制
app.state.max_content_length = 16 * 1024 * 1024  # 16MB


# 多摄像头管理器
class MultiCameraManager:
    def __init__(self):
        self.cameras = {}  # 存储多个摄像头实例
        self.lock = asyncio.Lock() # 用于控制并发访问

    def add_camera(self, camera_id: str, rtsp_url: str):
        """添加摄像头"""
        if camera_id not in self.cameras:
            self.cameras[camera_id] = {
                'cap': None,
                'is_running': False,
                'frame_cache': None,
                'last_frame_time': 0,
                'clients': set(),
                'rtsp_url': rtsp_url
            }

    def remove_camera(self, camera_id: str):
        """移除摄像头"""
        if camera_id in self.cameras:
            self.stop_camera(camera_id)
            del self.cameras[camera_id]

    async def start_camera(self, camera_id: str):
        """启动指定摄像头"""
        if camera_id not in self.cameras:
            return False

        camera_info = self.cameras[camera_id]
        if camera_info['is_running'] and camera_info['cap'] and camera_info['cap'].isOpened():
            return True

        try:
            # 创建视频捕获对象
            cap = cv2.VideoCapture(camera_info['rtsp_url'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                camera_info['cap'] = cap
                camera_info['is_running'] = True
                print(f"摄像头 {camera_id} 连接已启动")
                # 启动帧捕获任务
                asyncio.create_task(self.capture_frames(camera_id))
                return True
            else:
                print(f"无法连接到摄像头 {camera_id}")
                return False
        except Exception as e:
            print(f"启动摄像头 {camera_id} 时出错: {e}")
            return False

    async def capture_frames(self, camera_id: str):
        """持续捕获指定摄像头的帧"""
        if camera_id not in self.cameras:
            return

        camera_info = self.cameras[camera_id]
        while camera_info['is_running'] and camera_info['cap'] and camera_info['cap'].isOpened():
            try:
                ret, frame = camera_info['cap'].read()
                if ret:
                    camera_info['frame_cache'] = frame.copy()
                    camera_info['last_frame_time'] = time.time()
                await asyncio.sleep(0.03)  # 约30 FPS
            except Exception as e:
                print(f"捕获摄像头 {camera_id} 帧时出错: {e}")
                await asyncio.sleep(1)

    def get_frame(self, camera_id: str):
        """获取指定摄像头的最新帧"""
        if camera_id in self.cameras:
            return self.cameras[camera_id]['frame_cache']
        return None

    def add_client(self, camera_id: str, websocket):
        """为指定摄像头添加客户端"""
        if camera_id in self.cameras:
            self.cameras[camera_id]['clients'].add(websocket)

    def remove_client(self, camera_id: str, websocket):
        """为指定摄像头移除客户端"""
        if camera_id in self.cameras:
            self.cameras[camera_id]['clients'].discard(websocket)

    def stop_camera(self, camera_id: str):
        """停止指定摄像头"""
        if camera_id in self.cameras:
            camera_info = self.cameras[camera_id]
            camera_info['is_running'] = False
            if camera_info['cap']:
                camera_info['cap'].release()
            camera_info['clients'].clear()
            print(f"摄像头 {camera_id} 连接已停止")

    def get_camera_status(self, camera_id: str):
        """获取指定摄像头状态"""
        if camera_id in self.cameras:
            camera_info = self.cameras[camera_id]
            return {
                "is_running": camera_info['is_running'],
                "client_count": len(camera_info['clients']),
                "last_frame_time": camera_info['last_frame_time']
            }
        return None

    def get_all_status(self):
        """获取所有摄像头状态"""
        status = {}
        for camera_id in self.cameras:
            status[camera_id] = self.get_camera_status(camera_id)
        return status

    @app.websocket("/ws/camera_stream/{camera_id}")
    async def websocket_camera_stream(websocket: WebSocket, camera_id: str):
        """
        WebSocket实时摄像头检测接口（优化版）
        """
        await websocket.accept()
        print(f"WebSocket 客户端已连接到摄像头 {camera_id}")

        # 检查摄像头是否存在
        if multi_camera_manager.get_camera_status(camera_id) is None:
            # 如果摄像头不存在，尝试添加并启动它
            rtsp_urls = {
                "camera1": "rtsp://admin:123456@192.168.3.123:554/stream1",
                "camera2": "rtsp://admin:123456@192.168.3.124:554/stream1",
                "camera3": "rtsp://admin:123456@192.168.3.125:554/stream1"
            }
            rtsp_url = rtsp_urls.get(camera_id, f"rtsp://admin:123456@192.168.3.{123 + int(camera_id[-1])}:554/stream1")
            multi_camera_manager.add_camera(camera_id, rtsp_url)

            # 尝试启动摄像头
            success = await multi_camera_manager.start_camera(camera_id)
            if not success:
                try:
                    await websocket.send_text(json.dumps({
                        "error": f"无法启动摄像头 {camera_id}"
                    }))
                except:
                    pass
                await websocket.close()
                return

        # 添加客户端到管理器
        multi_camera_manager.add_client(camera_id, websocket)

        # 确保摄像头已启动
        camera_status = multi_camera_manager.get_camera_status(camera_id)
        if not camera_status or not camera_status['is_running']:
            success = await multi_camera_manager.start_camera(camera_id)
            if not success:
                try:
                    await websocket.send_text(json.dumps({
                        "error": f"无法启动摄像头 {camera_id}"
                    }))
                except:
                    pass
                multi_camera_manager.remove_client(camera_id, websocket)
                await websocket.close()
                return

        # 优化参数
        frame_count = 0
        last_process_time = 0
        process_interval = 0.1 # 每100ms处理一次（10 FPS）0.4
        quality = 60  # 降低图像质量以减少传输时间

        try:
            while True:
                current_time = time.time()

                # 获取最新帧
                frame = multi_camera_manager.get_frame(camera_id)
                if frame is not None:
                    frame_count += 1

                    # 控制处理频率 - 每100ms处理一次
                    if current_time - last_process_time >= process_interval:
                        last_process_time = current_time

                        # 调整图像大小以提高处理速度
                        height, width = frame.shape[:2]
                        # 降低分辨率以提高处理速度
                        target_width = 640
                        if width > target_width:
                            scale = target_width / width
                            new_width = target_width
                            new_height = int(height * scale)
                            frame_resized = cv2.resize(frame, (new_width, new_height))
                        else:
                            frame_resized = frame

                        # 使用 YOLO 模型进行推理
                        try:
                            # 使用更低的置信度阈值以减少处理时间
                            # 在模型推理时使用更快的设置
                            results = model(frame_resized, device='cuda:0' if torch.cuda.is_available() else 'cpu',
                                            conf=0.4, iou=0.45, max_det=10)

                            # 解析推理结果
                            detections = []
                            detection_count = 0

                            if hasattr(results[0], 'boxes'):
                                # 限制检测对象数量以提高性能
                                for i, box in enumerate(results[0].boxes.xyxy):
                                    if detection_count >= 10:  # 最多处理10个对象
                                        break

                                    x1, y1, x2, y2 = map(int, box)
                                    conf = float(results[0].boxes.conf[i])
                                    cls = int(results[0].boxes.cls[i])
                                    label = model.names[cls]

                                    if conf > 0.4:  # 置信度阈值
                                        detections.append({
                                            'label': label,
                                            'confidence': conf,
                                            'class': cls
                                        })
                                        detection_count += 1

                            # 生成标注图像（如果需要）
                            if detections:  # 只有检测到对象时才生成标注图像
                                annotated_frame = results[0].plot()
                            else:
                                annotated_frame = frame_resized

                            # 优化编码参数
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
                            processed_image_data = base64.b64encode(buffer).decode('utf-8')

                            # 发送结果到前端
                            response_data = {
                                'camera_id': camera_id,
                                'detections': detections,
                                'processed_image': f"data:image/jpeg;base64,{processed_image_data}",
                                'frame_count': frame_count,
                                'timestamp': current_time
                            }

                            # 发送前检查连接状态
                            try:
                                await websocket.send_text(json.dumps(response_data))
                            except Exception as send_error:
                                print(f"发送数据失败: {send_error}")
                                break

                        except Exception as e:
                            error_msg = f"检测过程中出错: {str(e)}"
                            print(error_msg)
                            try:
                                await websocket.send_text(json.dumps({"error": error_msg}))
                            except:
                                break

                    # 更短的等待时间以提高响应性
                    await asyncio.sleep(0.01)  # 10ms
                else:
                    # 没有帧数据，短暂等待
                    await asyncio.sleep(0.03)  # 30ms

        except Exception as e:
            error_msg = f"处理摄像头流时出错: {str(e)}"
            print(error_msg)
            try:
                await websocket.send_text(json.dumps({"error": error_msg}))
            except:
                pass
        finally:
            # 移除客户端
            multi_camera_manager.remove_client(camera_id, websocket)
            try:
                await websocket.close()
            except:
                pass
            print(f"WebSocket 客户端已断开连接到摄像头 {camera_id}")
            multi_camera_manager.stop_camera(camera_id)
            print(f"已停止摄像头 {camera_id}")


# 全局多摄像头管理器实例
multi_camera_manager = MultiCameraManager()

# 加载 YOLO 模型
model = YOLO("./yolo11n.pt")


# 添加新的数据模型
class CameraCaptureRequest(BaseModel):
    """摄像头捕获请求模型"""
    camera_id: str = "camera1"  # 指定摄像头ID
    quality: int = 60  # JPEG 质量 (1-100)
    resize_width: int = None  # 调整宽度
    confidence_threshold: float = 0.5  # 置信度阈值


class ImageRequest(BaseModel):
    """图片请求模型"""
    image: str  # Base64编码的图片数据

# 在应用启动时初始化摄像头
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化摄像头"""
    # 添加3个摄像头（请根据实际情况修改RTSP地址）
    multi_camera_manager.add_camera("camera1", "rtsp://admin:123456@192.168.3.123:554/stream1")
    multi_camera_manager.add_camera("camera2", "rtsp://admin:123456@192.168.3.124:554/stream1")
    multi_camera_manager.add_camera("camera3", "rtsp://admin:123456@192.168.3.125:554/stream1")

    # 启动所有摄像头
    # for camera_id in ["camera1", "camera2", "camera3"]:
    #     await multi_camera_manager.start_camera(camera_id)
    print("摄像头已初始化，但未启动")
# 在应用启动时初始化摄像头
# @app.on_event("startup")
# async def startup_event():
#     """应用启动时初始化摄像头"""
#     # 只添加和启动实际存在的摄像头
#     multi_camera_manager.add_camera("camera3", "rtsp://admin:123456@192.168.3.125:554/stream1")
#
#     # 只启动实际存在的摄像头
#     success = await multi_camera_manager.start_camera("camera3")
#     if success:
#         print("摄像头 camera3 启动成功")
#     else:
#         print("摄像头 camera3 启动失败")


# =============================================================================
# API 接口定义
# =============================================================================

@app.post("/camera/capture/{camera_id}",
          summary="摄像头捕获检测",
          description="捕获指定摄像头当前帧并进行目标检测，返回检测结果和标注图像")
async def capture_camera_frame(camera_id: str, request: CameraCaptureRequest = None):
    """
    捕获指定摄像头当前帧并进行目标检测

    - **camera_id**: 摄像头ID (路径参数)
    - **quality**: JPEG图像质量 (1-100)，默认80
    - **resize_width**: 调整图像宽度，保持宽高比
    - **confidence_threshold**: 检测置信度阈值 (0.0-1.0)，默认0.5

    返回:
    - **success**: 操作是否成功
    - **camera_id**: 摄像头ID
    - **detections**: 检测到的对象列表
    - **processed_image**: Base64编码的检测结果图像
    - **timestamp**: 时间戳
    - **detection_count**: 检测到的对象数量
    """
    if request is None:
        request = CameraCaptureRequest()

    # 设置请求中的 camera_id 为路径参数值
    request.camera_id = camera_id

    # 检查摄像头是否存在
    camera_status = multi_camera_manager.get_camera_status(request.camera_id)
    if camera_status is None:
        raise HTTPException(status_code=404, detail=f"摄像头 {request.camera_id} 不存在")

    # 获取摄像头帧
    frame = multi_camera_manager.get_frame(request.camera_id)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"摄像头 {request.camera_id} 未准备就绪，请稍后再试")

    try:
        # 调整图像尺寸（如果指定）
        if request.resize_width and request.resize_width > 0:
            height, width = frame.shape[:2]
            new_width = min(request.resize_width, width)
            new_height = int(height * (new_width / width))
            frame = cv2.resize(frame, (new_width, new_height))

        # 使用 YOLO 模型进行推理
        try:
            results = model(frame, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")

        # 解析推理结果
        detections = []
        if hasattr(results[0], 'boxes'):
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                conf = float(results[0].boxes.conf[i])
                cls = int(results[0].boxes.cls[i])
                label = model.names[cls]

                # 根据置信度阈值过滤结果
                if conf >= request.confidence_threshold:
                    detections.append({
                        'label': label,
                        'confidence': conf,
                        'class': cls,
                        'bbox': [x1, y1, x2, y2]
                    })

        # 生成标注图像
        annotated_frame = results[0].plot()

        # 编码为 base64
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), request.quality]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        processed_image_data = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            'success': True,
            'camera_id': request.camera_id,
            'detections': detections,
            'processed_image': f"data:image/jpeg;base64,{processed_image_data}",
            'timestamp': time.time(),
            'detection_count': len(detections)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/camera/raw/{camera_id}",
         summary="获取指定摄像头原始图像",
         description="获取指定摄像头原始图像（不进行目标检测），返回Base64编码的图像")
async def get_raw_camera_frame(
        camera_id: str,
        quality: int = 60,
        width: int = None
):
    """
    获取指定摄像头原始图像（不进行目标检测）

    - **camera_id**: 摄像头ID
    - **quality**: JPEG图像质量 (1-100)，默认80
    - **width**: 调整图像宽度，保持宽高比

    返回:
    - **success**: 操作是否成功
    - **camera_id**: 摄像头ID
    - **image**: Base64编码的原始摄像头图像
    - **timestamp**: 时间戳
    """
    # 检查摄像头是否存在
    camera_status = multi_camera_manager.get_camera_status(camera_id)
    if camera_status is None:
        raise HTTPException(status_code=404, detail=f"摄像头 {camera_id} 不存在")

    # 获取摄像头帧
    frame = multi_camera_manager.get_frame(camera_id)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"摄像头 {camera_id} 未准备就绪")

    try:
        # 调整尺寸
        if width and width > 0:
            height, original_width = frame.shape[:2]
            new_width = min(width, original_width)
            new_height = int(height * (new_width / original_width))
            frame = cv2.resize(frame, (new_width, new_height))

        # 编码为 base64
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        image_data = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            'success': True,
            'camera_id': camera_id,
            'image': f"data:image/jpeg;base64,{image_data}",
            'timestamp': time.time()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像处理失败: {str(e)}")


@app.get("/camera/status/{camera_id}",
         summary="获取指定摄像头状态",
         description="检查指定摄像头是否准备就绪，返回摄像头状态信息")
async def get_camera_status(camera_id: str):
    """
    获取指定摄像头状态

    - **camera_id**: 摄像头ID

    返回:
    - **camera_id**: 摄像头ID
    - **camera_ready**: 摄像头是否准备就绪
    - **timestamp**: 时间戳
    """
    camera_status = multi_camera_manager.get_camera_status(camera_id)
    if camera_status is None:
        raise HTTPException(status_code=404, detail=f"摄像头 {camera_id} 不存在")

    return {
        'camera_id': camera_id,
        'camera_ready': camera_status['is_running'],
        'timestamp': time.time()
    }


@app.get("/camera/status",
         summary="获取所有摄像头状态",
         description="检查所有摄像头是否准备就绪，返回摄像头状态信息")
async def get_all_cameras_status():
    """
    获取所有摄像头状态

    返回:
    - **cameras**: 所有摄像头状态列表
    """
    all_status = multi_camera_manager.get_all_status()
    return {
        'cameras': all_status,
        'timestamp': time.time()
    }


@app.get("/",
         summary="根目录访问",
         description="返回主页面HTML文件或欢迎信息")
async def read_root():
    """
    根路径访问

    返回:
    - **detection.html** 文件或欢迎消息
    """
    if os.path.exists("detection.html"):
        return FileResponse("detection.html")
    return {"message": "Welcome to YOLO Object Detection API"}


@app.get("/detection.html",
         summary="访问检测页面",
         description="直接访问检测HTML页面")
async def read_detection_html():
    """
    直接访问 detection.html

    返回:
    - **detection.html** 文件或错误信息
    """
    if os.path.exists("detection.html"):
        return FileResponse("detection.html")
    return {"error": "File not found"}


@app.post("/test_post",
          summary="测试POST请求",
          description="测试POST请求功能")
async def test_post(request: ImageRequest):
    """
    测试POST请求功能

    - **request**: ImageRequest对象

    返回:
    - **message**: 测试成功消息
    - **received_image_length**: 图像数据长度
    """
    return {"message": "POST 请求成功", "received_image_length": len(request.image)}


@app.get("/test_get",
         summary="测试GET请求",
         description="测试GET请求功能")
async def test_get():
    """
    测试GET请求功能

    返回:
    - **message**: 测试成功消息
    """
    return {"message": "GET 请求成功"}


@app.post("/infer",
          summary="图片上传检测",
          description="接收上传的图片进行目标检测，返回检测结果和标注图像")
async def infer(request: ImageRequest):
    """
    图片上传检测接口

    - **request**: ImageRequest对象，包含Base64编码的图像数据

    返回:
    - **inference_results**: 检测结果列表
    - **processed_image**: Base64编码的标注图像
    """
    try:
        base64_image = request.image

        # 处理 MIME 类型前缀
        if base64_image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,',
                                    'data:image/gif;base64,', 'data:image/bmp;base64,',
                                    'data:image/webp;base64,')):
            base64_image = base64_image.split(',')[1]

        # 解码 Base64 图片
        try:
            image_data = base64.b64decode(base64_image)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Failed to decode image: Image is empty")

        # 使用 YOLO 模型进行推理
        try:
            results = model(image, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

        # 解析推理结果
        detections = []
        if hasattr(results[0], 'boxes'):
            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                conf = float(results[0].boxes.conf[i])
                cls = int(results[0].boxes.cls[i])
                label = model.names[cls]
                detections.append({
                    'label': label,
                    'confidence': conf,
                    'class': cls
                })

        # 生成标注图像
        test = results[0].plot()

        try:
            _, buffer = cv2.imencode('.jpg', test)
            processed_image_data = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")

        return JSONResponse({
            'inference_results': detections,
            'processed_image': f"data:image/jpeg;base64,{processed_image_data}"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/camera/start/{camera_id}",
         summary="启动指定摄像头",
         description="启动指定摄像头连接")
async def start_camera(camera_id: str):
    """
    启动指定摄像头

    - **camera_id**: 摄像头ID

    返回:
    - **status**: 启动状态
    - **message**: 状态消息
    """
    # 如果摄像头不存在，添加它
    if multi_camera_manager.get_camera_status(camera_id) is None:
        # 这里需要根据camera_id设置对应的RTSP地址
        rtsp_urls = {
            "camera1": "rtsp://admin:123456@192.168.3.123:554/stream1",
            "camera2": "rtsp://admin:123456@192.168.3.124:554/stream1",
            "camera3": "rtsp://admin:123456@192.168.3.125:554/stream1"
        }
        rtsp_url = rtsp_urls.get(camera_id, f"rtsp://admin:123456@192.168.3.{123 + int(camera_id[-1])}:554/stream1")
        multi_camera_manager.add_camera(camera_id, rtsp_url)

    success = await multi_camera_manager.start_camera(camera_id)
    if success:
        return {"status": "success", "message": f"摄像头 {camera_id} 已启动"}
    else:
        return {"status": "error", "message": f"无法启动摄像头 {camera_id}"}

@app.get("/camera/delete/{camera_id}",
         summary="删除指定摄像头",
         description="删除指定摄像头连接")
async def delete_camera(camera_id: str):
    """
    删除指定摄像头

    - **camera_id**: 摄像头ID

    返回:
    - **status**: 删除状态
    - **message**: 状态消息
    """
    multi_camera_manager.remove_camera(camera_id)
    return {"status": "success", "message": f"摄像头 {camera_id} 已删除"}
@app.get("/camera/stop/{camera_id}",
         summary="停止指定摄像头",
         description="手动停止指定摄像头连接")
async def stop_camera(camera_id: str):
    """
    手动停止指定摄像头

    - **camera_id**: 摄像头ID

    返回:
    - **status**: 停止状态
    - **message**: 状态消息
    """
    multi_camera_manager.stop_camera(camera_id)
    return {"status": "success", "message": f"摄像头 {camera_id} 已停止"}


@app.websocket("/ws/camera_stream/{camera_id}")
async def websocket_camera_stream(websocket: WebSocket, camera_id: str):
    """
    WebSocket实时摄像头检测接口（优化版）
    """
    await websocket.accept()
    print(f"WebSocket 客户端已连接到摄像头 {camera_id}")

    # 检查摄像头是否存在
    if multi_camera_manager.get_camera_status(camera_id) is None:
        # 如果摄像头不存在，尝试添加并启动它
        rtsp_urls = {
            "camera1": "rtsp://admin:123456@192.168.3.123:554/stream1",
            "camera2": "rtsp://admin:123456@192.168.3.124:554/stream1",
            "camera3": "rtsp://admin:123456@192.168.3.125:554/stream1"
        }
        rtsp_url = rtsp_urls.get(camera_id, f"rtsp://admin:123456@192.168.3.{123 + int(camera_id[-1])}:554/stream1")
        multi_camera_manager.add_camera(camera_id, rtsp_url)

        # 尝试启动摄像头
        success = await multi_camera_manager.start_camera(camera_id)
        if not success:
            try:
                await websocket.send_text(json.dumps({
                    "error": f"无法启动摄像头 {camera_id}"
                }))
            except:
                pass
            await websocket.close()
            return

    # 添加客户端到管理器
    multi_camera_manager.add_client(camera_id, websocket)

    # 确保摄像头已启动
    camera_status = multi_camera_manager.get_camera_status(camera_id)
    if not camera_status or not camera_status['is_running']:
        success = await multi_camera_manager.start_camera(camera_id)
        if not success:
            try:
                await websocket.send_text(json.dumps({
                    "error": f"无法启动摄像头 {camera_id}"
                }))
            except:
                pass
            multi_camera_manager.remove_client(camera_id, websocket)
            await websocket.close()
            return

    # 优化参数
    frame_count = 0
    last_process_time = 0
    process_interval = 0.1  # 每100ms处理一次（10 FPS）
    quality = 60  # 降低图像质量以减少传输时间

    try:
        while True:
            current_time = time.time()

            # 获取最新帧
            frame = multi_camera_manager.get_frame(camera_id)
            if frame is not None:
                frame_count += 1

                # 控制处理频率 - 每100ms处理一次
                if current_time - last_process_time >= process_interval:
                    last_process_time = current_time

                    # 调整图像大小以提高处理速度
                    height, width = frame.shape[:2]
                    # 降低分辨率以提高处理速度
                    target_width = 640
                    if width > target_width:
                        scale = target_width / width
                        new_width = target_width
                        new_height = int(height * scale)
                        frame_resized = cv2.resize(frame, (new_width, new_height))
                    else:
                        frame_resized = frame

                    # 使用 YOLO 模型进行推理
                    try:
                        # 使用更低的置信度阈值以减少处理时间
                        results = model(frame_resized, device='cuda:0' if torch.cuda.is_available() else 'cpu',
                                        conf=0.4)

                        # 解析推理结果
                        detections = []
                        detection_count = 0

                        if hasattr(results[0], 'boxes'):
                            # 限制检测对象数量以提高性能
                            for i, box in enumerate(results[0].boxes.xyxy):
                                if detection_count >= 10:  # 最多处理10个对象
                                    break

                                x1, y1, x2, y2 = map(int, box)
                                conf = float(results[0].boxes.conf[i])
                                cls = int(results[0].boxes.cls[i])
                                label = model.names[cls]

                                if conf > 0.4:  # 置信度阈值
                                    detections.append({
                                        'label': label,
                                        'confidence': conf,
                                        'class': cls
                                    })
                                    detection_count += 1

                        # 生成标注图像（如果需要）
                        if detections:  # 只有检测到对象时才生成标注图像
                            annotated_frame = results[0].plot()
                        else:
                            annotated_frame = frame_resized

                        # 优化编码参数
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
                        processed_image_data = base64.b64encode(buffer).decode('utf-8')

                        # 发送结果到前端
                        response_data = {
                            'camera_id': camera_id,
                            'detections': detections,
                            'processed_image': f"data:image/jpeg;base64,{processed_image_data}",
                            'frame_count': frame_count,
                            'timestamp': current_time
                        }

                        # 发送前检查连接状态
                        try:
                            await websocket.send_text(json.dumps(response_data))
                        except Exception as send_error:
                            print(f"发送数据失败: {send_error}")
                            break

                    except Exception as e:
                        error_msg = f"检测过程中出错: {str(e)}"
                        print(error_msg)
                        try:
                            await websocket.send_text(json.dumps({"error": error_msg}))
                        except:
                            break

                # 更短的等待时间以提高响应性
                await asyncio.sleep(0.01)  # 10ms
            else:
                # 没有帧数据，短暂等待
                await asyncio.sleep(0.03)  # 30ms

    except Exception as e:
        error_msg = f"处理摄像头流时出错: {str(e)}"
        print(error_msg)
        try:
            await websocket.send_text(json.dumps({"error": error_msg}))
        except:
            pass
    finally:
        # 移除客户端
        multi_camera_manager.remove_client(camera_id, websocket)
        try:
            await websocket.close()
        except:
            pass
        print(f"WebSocket 客户端已断开连接到摄像头 {camera_id}")


@app.get("/test_camera/{camera_id}",
         summary="测试指定摄像头连接",
         description="测试指定RTSP摄像头连接是否正常")
async def test_camera(camera_id: str):
    """
    测试指定摄像头连接

    - **camera_id**: 摄像头ID

    返回:
    - **status**: 测试状态 ("success" 或 "error")
    - **message**: 测试结果消息
    """
    # 获取摄像头RTSP地址
    rtsp_urls = {
        "camera1": "rtsp://admin:123456@192.168.3.123:554/stream1",
        "camera2": "rtsp://admin:123456@192.168.3.124:554/stream1",
        "camera3": "rtsp://admin:123456@192.168.3.125:554/stream1"
    }
    rtsp_url = rtsp_urls.get(camera_id, f"rtsp://admin:123456@192.168.3.{123 + int(camera_id[-1])}:554/stream1")
# 使用OpenCV创建一个视频捕获对象，用于连接和读取RTSP视频流。
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        # 获取是否获取到视频帧
        ret, frame = cap.read()
        cap.release()
        if ret:
            return {"status": "success", "message": f"摄像头 {camera_id} 连接成功"}
        else:
            return {"status": "error", "message": f"摄像头 {camera_id} 无法读取视频帧"}
    else:
        return {"status": "error", "message": f"摄像头 {camera_id} 无法连接到摄像头"}


@app.get("/health",
         summary="健康检查",
         description="检查服务运行状态")
async def health_check():
    """
    健康检查接口

    返回:
    - **status**: 服务状态 ("healthy")
    - **message**: 状态消息
    """
    return {"status": "healthy", "message": "服务运行正常"}

# 3. 添加自定义的文档路由（在文件末尾添加，但在 if __name__ == '__main__': 之前）
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title + " - Swagger UI",
        oauth2_redirect_url="/swagger-ui/oauth2-redirect",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

# OAuth2重定向处理
@app.get("/swagger-ui/oauth2-redirect", include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()

# 添加这个简单的文档页面
@app.get("/simple-docs", include_in_schema=False)
async def simple_docs():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .endpoint { 
                border: 1px solid #ddd; 
                margin: 10px 0; 
                padding: 15px; 
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .method { 
                display: inline-block; 
                padding: 3px 8px; 
                border-radius: 3px; 
                color: white; 
                font-weight: bold;
            }
            .get { background-color: #28a745; }
            .post { background-color: #007bff; }
            .websocket { background-color: #6f42c1; }
        </style>
    </head>
    <body>
        <h1>YOLO Object Detection API</h1>
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/camera/status</strong> - 检查摄像头状态
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/camera/capture</strong> - 摄像头捕获检测
        </div>
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/camera/raw/{camera_id}</strong> - 获取原始摄像头图像
        </div>
        <div class="endpoint">
            <span class="method websocket">WebSocket</span>
            <strong>/ws/camera_stream/{camera_id}</strong> - 实时摄像头流
        </div>
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/infer</strong> - 图片上传检测
        </div>
    </body>
    </html>
    """)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
