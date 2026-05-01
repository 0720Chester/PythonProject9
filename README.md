# Whitted-Style 光线追踪渲染器

这是一个基于 Taichi 语言实现的 Whitted 风格光线追踪渲染器，支持多种材质和实时交互控制。

## 功能特性

- **多种材质类型**
  - 漫反射材质（红色球体）
  - 镜面反射材质（银色球体）
  - 棋盘格材质（地面平面）

- **光照模型**
  - Phong 光照模型
  - 阴影检测
  - 环境光、漫反射、镜面高光

- **交互控制**
  - 实时调整光源位置（X, Y, Z）
  - 实时调整最大光线弹射次数（1-5次）

## 场景构成

| 物体 | 类型 | 位置 | 材质 |
|------|------|------|------|
| 左球体 | 球体 | (-1.5, 0, 0) | 红色漫反射 |
| 右球体 | 球体 | (1.5, 0, 0) | 银色镜面 |
| 地面 | 无限大平面 | y = -1.0 | 棋盘格纹理 |

## 渲染参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 窗口分辨率 | 800×600 | 渲染输出尺寸 |
| 相机位置 | (0, 2, 8) | 观察点位置 |
| 视场角 | 60° | 相机视野角度 |
| 光源位置 | (3, 5, 2) | 可交互调整 |
| 光源颜色 | 白色 (1,1,1) | 强度为1 |
| 最大弹射次数 | 3（可调） | 光线追踪深度 |

## 核心算法

### 1. 光线-物体求交

```python
# 球体求交：解二次方程
intersect_sphere(ray_origin, ray_dir, center, radius)

# 平面求交：解线性方程
intersect_plane(ray_origin, ray_dir, plane_y)
```

### 2. Whitted 递归追踪

- 漫反射/棋盘格材质：直接计算光照后终止
- 镜面材质：生成反射光线，继续追踪
- 背景：生成渐变色天空

### 3. Phong 光照模型

```python
color = ambient + diffuse + specular
```

- 环境光系数：0.1
- 高光指数：32
- 高光强度：0.5

## 阴影计算

通过向光源方向发射阴影射线，检测路径上是否存在遮挡物：

```python
shadow_origin = hit_point + EPSILON * hit_normal + EPSILON * light_dir
shadow_t, ... = scene_intersect(shadow_origin, light_dir)
is_shadow = shadow_t < light_dist
```

## 运行方式

```bash
# 安装依赖
pip install taichi

# 运行渲染器
python ray_tracer.py
```

## 交互界面

运行后会显示一个 GUI 窗口，包含：
- **Light Position**：三个滑块控制光源 X, Y, Z 坐标
- **Max Bounces**：滑块控制最大反射次数（1-5）

## 技术栈

- **Taichi**：高性能 GPU 并行计算
- **Taichi.math**：向量和数学运算
- **Taichi.ui**：实时 GUI 交互

## 效果说明

- 红色漫反射球体：表现柔和的漫反射光照和阴影
- 银色镜面球体：表现清晰的反射效果，反射次数越多效果越真实
- 棋盘格地面：提供空间参考和阴影接收面
- 实时光源调整：可直观观察光照方向和阴影变化

## 代码结构

```
├── 材质常量定义
├── 几何工具函数
│   ├── reflect()
│   ├── intersect_sphere()
│   ├── intersect_plane()
│   └── scene_intersect()
├── 渲染函数
│   ├── is_in_shadow()
│   ├── get_checkerboard_color()
│   ├── phong_shading()
│   └── render() [kernel]


└── 主循环
    ├── GUI 控件
    └── 渲染循环
```

## 动图演示
<img width="800" height="450" alt="2026-05-0120-59-17-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/9c4c7322-7b2e-488a-83ef-b4cad5cc2c34" />
