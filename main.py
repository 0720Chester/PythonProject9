import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

# 窗口参数
WIDTH = 800
HEIGHT = 600

# 渲染参数
EPSILON = 1e-4
INF = 1e10

# 材质ID
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_CHECKERBOARD = 2

# 光源位置
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
light_color = ti.Vector.field(3, dtype=ti.f32, shape=())

# 最大弹射次数（使用taichi field以便在kernel中访问）
max_bounces = ti.field(dtype=ti.i32, shape=())

# 结果图像
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.func
def reflect(I, N):
    """计算反射向量"""
    return I - 2.0 * I.dot(N) * N


@ti.func
def intersect_sphere(ray_origin, ray_dir, center, radius):
    """光线与球体求交"""
    oc = ray_origin - center
    a = ray_dir.dot(ray_dir)
    b = 2.0 * oc.dot(ray_dir)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4.0 * a * c

    t = INF
    if discriminant >= 0.0:
        sqrt_d = ti.sqrt(discriminant)
        t0 = (-b - sqrt_d) / (2.0 * a)
        t1 = (-b + sqrt_d) / (2.0 * a)
        if t0 > EPSILON:
            t = t0
        else:
            if t1 > EPSILON:
                t = t1
    return t


@ti.func
def intersect_plane(ray_origin, ray_dir, plane_y):
    """光线与无限大平面求交 (y = plane_y)"""
    t = INF
    if ti.abs(ray_dir.y) > 1e-6:
        t_temp = (plane_y - ray_origin.y) / ray_dir.y
        if t_temp > EPSILON:
            t = t_temp
    return t


@ti.func
def scene_intersect(ray_origin, ray_dir):
    """场景求交，返回最近的交点信息"""
    closest_t = INF
    hit_material = MAT_DIFFUSE
    hit_point = tm.vec3(0.0)
    hit_normal = tm.vec3(0.0)

    # 无限大平面 y = -1.0
    t_plane = intersect_plane(ray_origin, ray_dir, -1.0)
    if t_plane < closest_t:
        closest_t = t_plane
        hit_point = ray_origin + t_plane * ray_dir
        hit_normal = tm.vec3(0.0, 1.0, 0.0)
        hit_material = MAT_CHECKERBOARD

    # 红色漫反射球 (左侧)
    center_diffuse = tm.vec3(-1.5, 0.0, 0.0)
    t_diffuse = intersect_sphere(ray_origin, ray_dir, center_diffuse, 1.0)
    if t_diffuse < closest_t:
        closest_t = t_diffuse
        hit_point = ray_origin + t_diffuse * ray_dir
        hit_normal = (hit_point - center_diffuse).normalized()
        hit_material = MAT_DIFFUSE

    # 银色镜面球 (右侧)
    center_mirror = tm.vec3(1.5, 0.0, 0.0)
    t_mirror = intersect_sphere(ray_origin, ray_dir, center_mirror, 1.0)
    if t_mirror < closest_t:
        closest_t = t_mirror
        hit_point = ray_origin + t_mirror * ray_dir
        hit_normal = (hit_point - center_mirror).normalized()
        hit_material = MAT_MIRROR

    return closest_t, hit_point, hit_normal, hit_material


@ti.func
def is_in_shadow(hit_point, hit_normal, light_dir, light_dist):
    """检测点是否在阴影中"""
    # 偏移起点避免自相交（沿法线方向和光线方向偏移）
    shadow_origin = hit_point + EPSILON * hit_normal + EPSILON * light_dir
    shadow_t, _, _, _ = scene_intersect(shadow_origin, light_dir)
    return shadow_t < light_dist


@ti.func
def get_checkerboard_color(point):
    """计算棋盘格颜色"""
    scale = 2.0
    ix = ti.cast(ti.floor(point.x * scale), ti.i32)
    iz = ti.cast(ti.floor(point.z * scale), ti.i32)
    color = tm.vec3(0.9, 0.9, 0.9)  # 默认白色
    if (ix + iz) % 2 != 0:
        color = tm.vec3(0.1, 0.1, 0.1)  # 黑色
    return color


@ti.func
def phong_shading(hit_point, hit_normal, view_dir, light_pos_val, light_color_val, diffuse_color):
    """Phong光照模型"""
    light_dir = light_pos_val - hit_point
    light_dist = light_dir.norm()
    light_dir = light_dir.normalized()

    # 环境光
    ambient = 0.1 * diffuse_color

    # 阴影检测
    color = ambient
    if not is_in_shadow(hit_point, hit_normal, light_dir, light_dist):
        # 漫反射
        diff = max(0.0, hit_normal.dot(light_dir))
        diffuse = diff * diffuse_color * light_color_val

        # 镜面高光
        reflect_dir = reflect(-light_dir, hit_normal)
        spec = max(0.0, view_dir.dot(reflect_dir))
        specular = ti.pow(spec, 32.0) * light_color_val * 0.5

        color = ambient + diffuse + specular

    return color


@ti.kernel
def render():
    """渲染主函数"""
    # 相机参数
    camera_pos = tm.vec3(0.0, 2.0, 8.0)
    camera_lookat = tm.vec3(0.0, 0.0, 0.0)
    camera_up = tm.vec3(0.0, 1.0, 0.0)
    fov = 60.0 * tm.pi / 180.0

    light_pos_val = light_pos[None]
    light_color_val = light_color[None]
    current_max_bounces = max_bounces[None]

    # 相机坐标系
    w = (camera_pos - camera_lookat).normalized()
    u = (camera_up.cross(w)).normalized()
    v = w.cross(u)

    for i, j in pixels:
        # 计算光线方向
        x = (ti.cast(i, ti.f32) + 0.5) / ti.cast(WIDTH, ti.f32) * 2.0 - 1.0
        y = (ti.cast(j, ti.f32) + 0.5) / ti.cast(HEIGHT, ti.f32) * 2.0 - 1.0
        aspect = ti.cast(WIDTH, ti.f32) / ti.cast(HEIGHT, ti.f32)

        ray_dir = (u * x * aspect * ti.tan(fov * 0.5) +
                   v * y * ti.tan(fov * 0.5) - w).normalized()

        # 迭代光线追踪
        ray_origin = camera_pos
        ray_direction = ray_dir
        throughput = tm.vec3(1.0, 1.0, 1.0)
        final_color = tm.vec3(0.0, 0.0, 0.0)

        for bounce in range(current_max_bounces):
            closest_t, hit_point, hit_normal, hit_material = scene_intersect(ray_origin, ray_direction)

            if closest_t >= INF:
                # 未击中任何物体，使用背景色（渐变天空）
                t = 0.5 * (ray_direction.y + 1.0)
                background = (1.0 - t) * tm.vec3(1.0, 1.0, 1.0) + t * tm.vec3(0.5, 0.7, 1.0)
                final_color += throughput * background
                break

            view_dir = -ray_direction

            if hit_material == MAT_CHECKERBOARD:
                diffuse_color = get_checkerboard_color(hit_point)
                color = phong_shading(hit_point, hit_normal, view_dir, light_pos_val, light_color_val, diffuse_color)
                final_color += throughput * color
                break

            elif hit_material == MAT_DIFFUSE:
                diffuse_color = tm.vec3(1.0, 0.2, 0.2)  # 红色
                color = phong_shading(hit_point, hit_normal, view_dir, light_pos_val, light_color_val, diffuse_color)
                final_color += throughput * color
                break

            else:  # MAT_MIRROR
                # 镜面反射
                reflect_dir = reflect(ray_direction, hit_normal)
                # 偏移起点避免自相交
                ray_origin = hit_point + EPSILON * hit_normal
                ray_direction = reflect_dir
                throughput *= 0.8  # 反射率

                # 如果这是最后一次弹射，添加环境光
                if bounce == current_max_bounces - 1:
                    ambient = 0.1 * throughput
                    final_color += ambient

        pixels[i, j] = tm.clamp(final_color, 0.0, 1.0)


def main():
    # 初始化光源
    light_pos[None] = tm.vec3(3.0, 5.0, 2.0)
    light_color[None] = tm.vec3(1.0, 1.0, 1.0)

    # 初始化最大弹射次数（使用taichi field）
    max_bounces[None] = 3

    # 创建窗口
    window = ti.ui.Window("Whitted-Style Ray Tracing", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    # 创建GUI
    gui = window.get_gui()

    # 渲染循环
    while window.running:
        # 设置UI控件
        gui.begin("Controls", 0.0, 0.0, 0.3, 0.4)
        gui.text("Light Position")
        light_x = light_pos[None].x
        light_y = light_pos[None].y
        light_z = light_pos[None].z

        new_x = gui.slider_float("Light X", light_x, -5.0, 5.0)
        if new_x != light_x:
            light_pos[None].x = new_x

        new_y = gui.slider_float("Light Y", light_y, 0.0, 10.0)
        if new_y != light_y:
            light_pos[None].y = new_y

        new_z = gui.slider_float("Light Z", light_z, -5.0, 5.0)
        if new_z != light_z:
            light_pos[None].z = new_z

        gui.text("")
        gui.text("Rendering")
        bounces = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)
        if bounces != max_bounces[None]:
            max_bounces[None] = bounces

        gui.end()

        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()
