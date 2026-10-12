# 智能屏幕支架：基于人脸追踪的机械臂自动屏幕姿态调整

## 项目背景

针对卧床、上肢受限人群无法自主调节屏幕角度的痛点，研发以小型机械臂为执行结构的智能屏幕支架。系统自动跟随用户脸部位置变化，实时调整屏幕距离与角度，始终保持最佳观看视角。

## 系统架构

```
RGB-D 相机
    │
    ▼
MediaPipe Face Mesh 人脸关键点检测 + 头部姿态估计
    │
    ▼
RGB-D 深度信息解算三维空间坐标
    │
    ▼
视觉坐标系 → 机械臂坐标系映射转换
    │
    ▼
低通滤波平滑运动指令
    │
    ▼
PiPER 机械臂调整屏幕位姿
```

## 核心模块

| 模块 | 文件 | 功能 |
|---|---|---|
| 人脸追踪 | `src/face_tracker.py` | MediaPipe Face Mesh 478 关键点，PnP 解算头部姿态 |
| 深度处理 | `src/depth_processor.py` | RGB-D 深度获取，像素坐标→三维坐标，虹膜测距兜底 |
| 坐标转换 | `src/coordinate_transform.py` | 相机坐标系→机械臂基坐标系，目标屏幕位姿计算 |
| 机械臂控制 | `src/arm_controller.py` | 速度限制、低通滤波、PiPER 机械臂指令 |
| 主流程 | `src/main.py` | 完整视觉-控制 pipeline |

## 测试结果

| 指标 | 结果 |
|---|---|
| 人脸检测准确率 | 98.2% |
| 支架响应延迟 | < 300ms |
| 测试场景 | 正常坐姿、卧床侧躺、弱光 |
| 运动表现 | 平顺无卡顿 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 仿真模式（无硬件）
python src/main.py --sim

# 仅视觉（不控制机械臂）
python src/main.py --no-arm

# 视频文件测试
python src/main.py --video test.mp4

# 完整运行（相机 + 机械臂）
python src/main.py
```

## 硬件配置

- **机械臂**：AgileX PiPER 6-DoF
- **相机**：Orbbec RGB-D（或普通 RGB 摄像头，虹膜测距兜底）
- **计算单元**：PC / 嵌入式主机

## 关键技术

- **MediaPipe Face Mesh**：478 个人脸关键点，含虹膜检测，实时头部姿态估计
- **PnP 位姿解算**：2D 关键点 + 3D 头部模型，输出 pitch/yaw/roll
- **RGB-D 三维重建**：深度数据 + 相机内参，像素坐标→三维空间坐标
- **坐标映射**：手眼标定，相机坐标系到机械臂坐标系的刚体变换
- **运动平滑**：指数低通滤波 + 速度限制，避免机械臂抖动

## 项目结构

```
smart-screen-stand/
├── src/
│   ├── main.py               # 主流程
│   ├── face_tracker.py       # 人脸追踪与头部姿态
│   ├── depth_processor.py    # 深度处理
│   ├── coordinate_transform.py  # 坐标转换
│   └── arm_controller.py     # 机械臂控制
├── config/
│   └── config.yaml           # 配置文件
├── tests/
├── docs/
└── requirements.txt
```

## 参考

- [MediaPipe Face Mesh](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [PiPER 机械臂](https://github.com/agilexrobotics)
- 基于 [PiPER Intelligent Workstation](https://github.com/smalleha/Adx_PiPER_Intelligent_workstation) 架构改进，将 YOLOv5-Face 替换为 MediaPipe Face Mesh
