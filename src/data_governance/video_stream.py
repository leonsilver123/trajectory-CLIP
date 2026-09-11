"""
src.data_governance.video_stream - 视频流管理模块

支持 RTSP 流和本地视频文件的统一接入:
- 每路流独立线程读取，帧缓冲区存储最新帧
- 断线自动重连
- 帧率控制
- 多路流状态监控

输入: RTSP 地址或本地视频文件路径
输出: numpy 图像帧 (BGR 格式)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np

from src.common.logger import get_logger

logger = get_logger("data_governance.video_stream")


class _StreamWorker:
    """
    单路视频流工作线程

    负责从一路视频源（RTSP 或本地文件）中持续读取帧，
    存入帧缓冲区，并在断线时自动重连。

    Attributes:
        camera_id: 摄像头唯一编号
        source: 视频源地址（RTSP URL 或文件路径）
        stream_type: 流类型 ("rtsp" 或 "file")
        target_fps: 目标帧率
        reconnect_interval: 断线重连间隔（秒）
        buffer_size: 帧缓冲区大小
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        stream_type: str,
        target_fps: float,
        reconnect_interval: float,
        buffer_size: int,
    ) -> None:
        """
        初始化单路流工作线程

        Args:
            camera_id: 摄像头唯一编号
            source: 视频源地址
            stream_type: 流类型 ("rtsp" / "file")
            target_fps: 目标读取帧率
            reconnect_interval: 断线重连等待秒数
            buffer_size: 帧缓冲区最大容量
        """
        self.camera_id = camera_id
        self.source = source
        self.stream_type = stream_type
        self.target_fps = target_fps
        self.reconnect_interval = reconnect_interval
        self.buffer_size = buffer_size

        # 帧缓冲区（保留最新 N 帧）
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        # 帧编号计数器
        self.frame_counter: int = 0
        # 线程控制
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        # 状态信息
        self.is_running: bool = False
        self.is_connected: bool = False
        self.last_error: Optional[str] = None
        self.last_frame_time: Optional[float] = None
        self.total_frames_read: int = 0
        self.reconnect_count: int = 0
        # VideoCapture 对象
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """启动读取线程"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"[{self.camera_id}] 读取线程已在运行")
            return

        self._stop_event.clear()
        self.is_running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"VideoStream-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[{self.camera_id}] 视频流读取线程已启动: {self.source}")

    def stop(self) -> None:
        """停止读取线程并释放资源"""
        self._stop_event.set()
        self.is_running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        self._release_capture()
        self.is_connected = False
        logger.info(f"[{self.camera_id}] 视频流已停止")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        获取缓冲区中最新的一帧

        Returns:
            最新帧图像 (BGR)，缓冲区为空时返回 None
        """
        with self._lock:
            if not self.frame_buffer:
                return None
            return self.frame_buffer[-1].copy()

    def get_frames(self, count: int) -> List[np.ndarray]:
        """
        从缓冲区中获取最近 count 帧

        Args:
            count: 需要获取的帧数

        Returns:
            帧列表（BGR），按时间正序排列，数量可能少于 count
        """
        with self._lock:
            frames = list(self.frame_buffer)
            # 取最近 count 帧
            frames = frames[-count:] if len(frames) > count else frames
            return [f.copy() for f in frames]

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前流的状态信息

        Returns:
            包含流状态的字典
        """
        return {
            "camera_id": self.camera_id,
            "source": self.source,
            "stream_type": self.stream_type,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "buffer_size": len(self.frame_buffer),
            "buffer_capacity": self.buffer_size,
            "total_frames_read": self.total_frames_read,
            "reconnect_count": self.reconnect_count,
            "last_frame_time": self.last_frame_time,
            "last_error": self.last_error,
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _connect(self) -> bool:
        """
        建立视频流连接

        Returns:
            连接是否成功
        """
        self._release_capture()

        try:
            # 根据流类型设置不同的 VideoCapture 参数
            if self.stream_type == "rtsp":
                # RTSP 流：设置 TCP 传输以提高稳定性
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            else:
                # 本地视频文件
                self._cap = cv2.VideoCapture(self.source)

            if self._cap is None or not self._cap.isOpened():
                logger.warning(f"[{self.camera_id}] 无法打开视频源: {self.source}")
                self.is_connected = False
                self.last_error = f"无法打开视频源: {self.source}"
                return False

            self.is_connected = True
            self.last_error = None
            logger.info(f"[{self.camera_id}] 视频流连接成功: {self.source}")
            return True

        except Exception as e:
            logger.error(f"[{self.camera_id}] 连接视频源异常: {e}")
            self.is_connected = False
            self.last_error = str(e)
            return False

    def _release_capture(self) -> None:
        """安全释放 VideoCapture 资源"""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _read_loop(self) -> None:
        """
        主读取循环（在工作线程中运行）

        持续从视频源读取帧，按 target_fps 控制帧率。
        连接断开后自动重连。
        """
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.1

        while not self._stop_event.is_set():
            # 确保已连接
            if not self.is_connected:
                if not self._connect():
                    logger.debug(
                        f"[{self.camera_id}] 等待 {self.reconnect_interval}s 后重试连接..."
                    )
                    # 等待重连间隔，期间检查停止信号
                    if self._stop_event.wait(timeout=self.reconnect_interval):
                        break
                    self.reconnect_count += 1
                    continue

            # 读取帧
            try:
                if self._cap is None:
                    self.is_connected = False
                    continue

                ret, frame = self._cap.read()

                if not ret or frame is None:
                    # 读取失败，可能是流结束或断线
                    if self.stream_type == "file":
                        # 本地文件读到末尾，从头开始
                        logger.info(f"[{self.camera_id}] 视频文件读取完毕，从头开始循环")
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # RTSP 流读取失败，需要重连
                        logger.warning(f"[{self.camera_id}] 帧读取失败，尝试重连")
                        self.is_connected = False
                        self.last_error = "帧读取失败"
                        self.reconnect_count += 1
                        if self._stop_event.wait(timeout=self.reconnect_interval):
                            break
                        continue

                # 将帧存入缓冲区
                with self._lock:
                    self.frame_buffer.append(frame)
                    self.frame_counter += 1
                    self.total_frames_read += 1
                    self.last_frame_time = time.time()

                # 帧率控制
                time.sleep(frame_interval)

            except Exception as e:
                logger.error(f"[{self.camera_id}] 读帧异常: {e}")
                self.is_connected = False
                self.last_error = str(e)
                self.reconnect_count += 1
                if self._stop_event.wait(timeout=self.reconnect_interval):
                    break


class VideoStreamManager:
    """
    视频流管理器 - 支持 RTSP 和本地视频文件

    统一管理多路视频流的接入、帧读取和状态监控。
    每路流分配独立线程，帧数据存入环形缓冲区。

    使用方式:
        config = {
            "streams": [
                {"camera_id": "c001", "source": "cityflow/.../train/S01/c001/vdo.avi", "type": "file"},
                {"camera_id": "c002", "source": "rtsp://...", "type": "rtsp"},
            ],
            "target_fps": 10,
            "reconnect_interval": 5,
            "buffer_size": 30,
        }
        manager = VideoStreamManager(config)
        manager.start_all()
        frame = manager.get_frame("c001")
        manager.stop_all()
    """

    def __init__(self, config: dict) -> None:
        """
        初始化视频流管理器

        Args:
            config: 视频流配置字典，格式:
                {
                    "streams": [
                        {"camera_id": "c001", "source": "path/to/vdo.avi", "type": "file"},
                        {"camera_id": "c002", "source": "rtsp://...", "type": "rtsp"},
                    ],
                    "target_fps": 10,          # 目标帧率（默认 10）
                    "reconnect_interval": 5,    # 断线重连间隔秒数（默认 5）
                    "buffer_size": 30           # 帧缓冲区大小（默认 30）
                }
        """
        self._workers: Dict[str, _StreamWorker] = {}
        self._lock = threading.Lock()

        # 解析全局参数
        target_fps = config.get("target_fps", 10)
        reconnect_interval = config.get("reconnect_interval", 5)
        buffer_size = config.get("buffer_size", 30)

        # 为每路流创建 Worker
        streams = config.get("streams", [])
        for stream_cfg in streams:
            camera_id = stream_cfg.get("camera_id")
            source = stream_cfg.get("source")
            stream_type = stream_cfg.get("type", "file")

            if not camera_id or not source:
                logger.warning(f"跳过无效流配置: {stream_cfg}")
                continue

            if camera_id in self._workers:
                logger.warning(f"摄像头 {camera_id} 重复配置，将覆盖之前的配置")

            worker = _StreamWorker(
                camera_id=camera_id,
                source=source,
                stream_type=stream_type,
                target_fps=target_fps,
                reconnect_interval=reconnect_interval,
                buffer_size=buffer_size,
            )
            self._workers[camera_id] = worker

        logger.info(
            f"视频流管理器初始化完成，共 {len(self._workers)} 路流，"
            f"target_fps={target_fps}, buffer_size={buffer_size}"
        )

    def start_all(self) -> None:
        """启动所有视频流的读取线程"""
        with self._lock:
            for camera_id, worker in self._workers.items():
                try:
                    worker.start()
                except Exception as e:
                    logger.error(f"[{camera_id}] 启动失败: {e}")

        logger.info(f"已启动 {len(self._workers)} 路视频流")

    def stop_all(self) -> None:
        """停止所有视频流并释放资源"""
        with self._lock:
            for camera_id, worker in self._workers.items():
                try:
                    worker.stop()
                except Exception as e:
                    logger.error(f"[{camera_id}] 停止失败: {e}")

        logger.info("所有视频流已停止")

    def start_stream(self, camera_id: str) -> bool:
        """
        启动指定摄像头的视频流

        Args:
            camera_id: 摄像头 ID

        Returns:
            是否成功启动
        """
        worker = self._workers.get(camera_id)
        if worker is None:
            logger.error(f"未知的摄像头 ID: {camera_id}")
            return False

        try:
            worker.start()
            return True
        except Exception as e:
            logger.error(f"[{camera_id}] 启动失败: {e}")
            return False

    def stop_stream(self, camera_id: str) -> bool:
        """
        停止指定摄像头的视频流

        Args:
            camera_id: 摄像头 ID

        Returns:
            是否成功停止
        """
        worker = self._workers.get(camera_id)
        if worker is None:
            logger.error(f"未知的摄像头 ID: {camera_id}")
            return False

        try:
            worker.stop()
            return True
        except Exception as e:
            logger.error(f"[{camera_id}] 停止失败: {e}")
            return False

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        获取指定摄像头的最新一帧

        Args:
            camera_id: 摄像头 ID

        Returns:
            最新帧图像 (BGR 格式, numpy array)，
            摄像头不存在或缓冲区为空时返回 None
        """
        worker = self._workers.get(camera_id)
        if worker is None:
            logger.warning(f"未知的摄像头 ID: {camera_id}")
            return None

        return worker.get_latest_frame()

    def get_frame_batch(self, camera_id: str, count: int) -> List[np.ndarray]:
        """
        批量获取指定摄像头的多帧

        从帧缓冲区中获取最近 count 帧，按时间正序排列。
        实际返回数量可能少于 count（缓冲区帧数不足时）。

        Args:
            camera_id: 摄像头 ID
            count: 需要获取的帧数

        Returns:
            帧列表 (BGR 格式)，按时间正序排列。
            摄像头不存在时返回空列表。
        """
        worker = self._workers.get(camera_id)
        if worker is None:
            logger.warning(f"未知的摄像头 ID: {camera_id}")
            return []

        return worker.get_frames(count)

    def get_status(self) -> Dict[str, dict]:
        """
        获取所有视频流的状态

        Returns:
            字典，key 为 camera_id，value 为该路流的状态字典，包含:
                - is_running: 是否在运行
                - is_connected: 是否已连接
                - buffer_size: 当前缓冲区帧数
                - total_frames_read: 累计读取帧数
                - reconnect_count: 重连次数
                - last_frame_time: 最后一帧的时间戳
                - last_error: 最近的错误信息
        """
        with self._lock:
            return {
                camera_id: worker.get_status()
                for camera_id, worker in self._workers.items()
            }

    def add_stream(
        self,
        camera_id: str,
        source: str,
        stream_type: str = "file",
        target_fps: float = 10,
        reconnect_interval: float = 5,
        buffer_size: int = 30,
    ) -> bool:
        """
        动态添加一路视频流

        Args:
            camera_id: 摄像头 ID
            source: 视频源地址
            stream_type: 流类型 ("rtsp" / "file")
            target_fps: 目标帧率
            reconnect_interval: 断线重连间隔秒数
            buffer_size: 帧缓冲区大小

        Returns:
            是否成功添加
        """
        with self._lock:
            if camera_id in self._workers:
                logger.warning(f"摄像头 {camera_id} 已存在，请先移除再添加")
                return False

            worker = _StreamWorker(
                camera_id=camera_id,
                source=source,
                stream_type=stream_type,
                target_fps=target_fps,
                reconnect_interval=reconnect_interval,
                buffer_size=buffer_size,
            )
            self._workers[camera_id] = worker
            worker.start()
            logger.info(f"动态添加视频流: {camera_id} -> {source}")
            return True

    def remove_stream(self, camera_id: str) -> bool:
        """
        动态移除一路视频流

        Args:
            camera_id: 摄像头 ID

        Returns:
            是否成功移除
        """
        with self._lock:
            worker = self._workers.pop(camera_id, None)
            if worker is None:
                logger.warning(f"摄像头 {camera_id} 不存在")
                return False

        worker.stop()
        logger.info(f"已移除视频流: {camera_id}")
        return True

    @property
    def stream_count(self) -> int:
        """当前管理的视频流数量"""
        return len(self._workers)

    @property
    def camera_ids(self) -> List[str]:
        """所有已注册的摄像头 ID 列表"""
        return list(self._workers.keys())

    @classmethod
    def from_aicity22(
        cls,
        scene: str,
        split: str = "train",
        target_fps: int = 10,
        buffer_size: int = 30,
    ) -> "VideoStreamManager":
        """
        从 AICity22 数据集场景创建视频流管理器

        自动构建该场景下所有摄像头的 vdo.avi 视频源。

        Args:
            scene: 场景 ID（如 'S01'）
            split: 数据划分 ('train'/'validation'/'test')
            target_fps: 目标帧率
            buffer_size: 帧缓冲区大小

        Returns:
            配置好的 VideoStreamManager 实例
        """
        from src.common.config import (
            DATASET_ROOT,
            SCENE_CAMERA_MAP,
            CAMERA_FPS,
        )

        cameras = SCENE_CAMERA_MAP.get(scene, [])
        streams = []
        for cam_id in cameras:
            video_path = f"{DATASET_ROOT}/{split}/{scene}/{cam_id}/vdo.avi"
            streams.append({
                "camera_id": cam_id,
                "source": video_path,
                "type": "file",
            })

        config = {
            "streams": streams,
            "target_fps": target_fps,
            "reconnect_interval": 5,
            "buffer_size": buffer_size,
        }
        return cls(config)
