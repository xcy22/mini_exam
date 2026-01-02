import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os  
from collections import defaultdict

# ===================== Core Configuration (Modify according to actual conditions) =====================
# 1. ArUco字典配置
ORIGINAL_ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
CAMERA_INDEX = 1
TAG_SIZE = 80.0  # ArUco标签物理尺寸(mm)

# 2. 摄像头内参+畸变系数
# CAMERA_MTX = np.array([[458.03440683,   0.,         329.88026673],
#                        [  0.,         458.09337598, 240.11466062],
#                        [  0.,           0.,           1.        ]], dtype=np.float64)
# CAMERA_DIST = np.array([-4.26069785e-01,  1.77533237e-01, -8.85825436e-04, -4.12779641e-04,
#    1.49360377e-01], dtype=np.float64)

#6128
CAMERA_MTX = np.array([[459.89003493,   0.,         348.02920611],
                       [  0.,         460.49332821, 242.05609682],
                       [  0.,           0.,           1.        ]])
CAMERA_DIST = np.array([-0.42210612,  0.24408866, -0.00129294,  0.00104831, -0.09170899])
# 去畸变裁剪参数
ALPHA = 0  # 0=裁剪无效区域并缩放对齐，1=保留黑边

# 3. 拍照保存配置
SAVE_FOLDER = "aruco_photos"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# 4. 紫色小球检测配置（OpenCV HSV格式）
# PURPLE_HSV_LOW = np.array([150, 100, 70])    
# PURPLE_HSV_HIGH = np.array([200, 140, 110]) 
# BALL_RADIUS_MIN = 10
# BALL_RADIUS_MAX = 25
# HOUGH_DP = 1.2
# HOUGH_MIN_DIST = 100
# HOUGH_PARAM1 = 25
# HOUGH_PARAM2 = 20
#undist setting
PURPLE_HSV_LOW = np.array([150, 100, 70])    # 您实测转换后的下限
PURPLE_HSV_HIGH = np.array([200, 155, 110]) # 您实测转换后的上限
BALL_RADIUS_MIN = 2
BALL_RADIUS_MAX = 10
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 100
HOUGH_PARAM1 = 10
HOUGH_PARAM2 = 10

# 5. 采集配置
TARGET_VALID_FRAMES = 3  # 目标有效帧数（收集到该数量才停止）
MAX_TOTAL_FRAMES = 500    # 最大总采集帧数（防止无限循环）

# 6. 真实世界坐标配置（点1-4的真实3D坐标，单位：mm）
REAL_WORLD_COORDS = {
    1: np.array([43.95, 17.40, 49.70]),  # 点1真实坐标 (x, y, z)
    2: np.array([33.95, 17.40, 49.62]),  # 点2真实坐标
    3: np.array([-34.90, 17.40, 49.08]), # 点3真实坐标
    4: np.array([-44.90, 17.40, 49.00])  # 点4真实坐标
}

# 7. NPY文件保存路径（点5真实坐标的存储路径）
NPY_SAVE_PATH = "data/point5_real_coordinates.npy"  # 可自定义路径/文件名

# ===================== Utility Functions =====================
def calculate_aruco_pose(tag_3d, tag_2d, mtx, dist):
    """计算ArUco位姿（保留，用于可视化）"""
    retval, rvec, tvec = cv2.solvePnP(tag_3d, tag_2d, mtx, dist)
    if not retval:
        return None, None
    return rvec, tvec

def detect_purple_ball(frame, hsv_low, hsv_high, r_min, r_max,
                       hough_dp, hough_min_dist, hough_param1, hough_param2):
    """检测紫色小球，返回第一个有效小球的中心坐标(x,y)，无则返回None"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    purple_region = cv2.bitwise_and(frame, frame, mask=mask)
    gray_purple = cv2.cvtColor(purple_region, cv2.COLOR_BGR2GRAY)
    gray_purple = cv2.GaussianBlur(gray_purple, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        gray_purple, cv2.HOUGH_GRADIENT, dp=hough_dp,
        minDist=hough_min_dist, param1=hough_param1,
        param2=hough_param2, minRadius=r_min, maxRadius=r_max
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle[0], circle[1], circle[2]
            if r_min < radius < r_max:
                # 绘制小球（可视化）
                cv2.circle(frame, (x, y), radius, (255, 0, 255), 2)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                return (x, y)  # 返回第一个有效小球坐标
    return None

def extract_and_sort_aruco_corners(corners_list, frame):
    """
    从两个ArUco码中提取第二/第三个角点（共4个），按X坐标排序并返回X列表
    :param corners_list: 两个ArUco码的角点列表 [corner_0, corner_1]，每个corner是(4,2)数组
    :param frame: 用于绘制角点编号的画面
    :return: [x1, x2, x3, x4] （X从小到大），失败返回None
    """
    # 确保传入两个ArUco码的角点
    if len(corners_list) < 2:
        return None
    
    # 提取第一个ArUco码的第二、第三个角点（索引1、2）
    aruco1_corner2 = corners_list[0][0][1]  # 第二个角点
    aruco1_corner3 = corners_list[0][0][2]  # 第三个角点
    # 提取第二个ArUco码的第二、第三个角点（索引1、2）
    aruco2_corner2 = corners_list[1][0][1]  # 第二个角点
    aruco2_corner3 = corners_list[1][0][2]  # 第三个角点
    
    # 收集这4个角点
    four_corners = [aruco1_corner2, aruco1_corner3, aruco2_corner2, aruco2_corner3]
    
    # 验证角点格式
    for corner in four_corners:
        if len(corner) != 2:
            return None
    
    # 按X坐标升序排序
    corner_with_x = [(corner[0], corner[1]) for corner in four_corners]
    corner_with_x.sort(key=lambda x: x[0])  # 按X坐标排序
    
    # 提取排序后的X坐标
    sorted_x = [round(corner[0]) for corner in corner_with_x]
    
    # 绘制排序后的角点（可视化：1-4号点，绿色）
    for i, (x, y) in enumerate(corner_with_x):
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)  # 绿色实心圆
        cv2.putText(frame, str(i+1), (int(x)+8, int(y)+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 编号
    
    return sorted_x

# ===================== Main Function: Frame Collection & Calculation =====================
def collect_frames_and_calculate():
    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
      

    if not cap.isOpened():
        print("❌ Error: Failed to open camera!")
        return

    # 1. 获取原始尺寸并计算去畸变参数（按照camera_undist.py的方法）
    ret, frame_raw = cap.read()
    if not ret:
        print("❌ Error: Failed to read frame for undistort init!")
        cap.release()
        return
    h_raw, w_raw = frame_raw.shape[:2]  # 原始画面尺寸（1920x1080）
    
    # 计算最优内参矩阵和ROI
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        CAMERA_MTX, CAMERA_DIST, (w_raw, h_raw), ALPHA, (w_raw, h_raw)
    )
    
    # 预计算去畸变映射表
    mapx, mapy = cv2.initUndistortRectifyMap(
        CAMERA_MTX, CAMERA_DIST, None, newcameramtx, (w_raw, h_raw), cv2.CV_32FC1
    )

    # 2. ArUco 3D坐标定义（用于位姿绘制）
    half_size = TAG_SIZE / 2
    aruco_3d_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float64)

    # 初始化 ArUco 检测器 (适配 OpenCV 4.7+)
    aruco_params = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(ORIGINAL_ARUCO_DICT, aruco_params)

    # 存储有效帧数据：key=点编号(1-5), value=X坐标列表
    valid_frame_data = defaultdict(list)
    collected_frame_count = 0  # 总采集帧数
    valid_frame_count = 0      # 有效帧数

    # 初始提示
    print(f"✅ Camera started! Collecting until {TARGET_VALID_FRAMES} valid frames (max {MAX_TOTAL_FRAMES} total frames)...")
    print("   Valid frame condition: 2 ArUco markers + purple ball detected")
    print("   Press 'q' to stop early")

    # 核心循环：收集到20个有效帧 或 达到最大帧数停止
    while valid_frame_count < TARGET_VALID_FRAMES and collected_frame_count < MAX_TOTAL_FRAMES:
        ret, frame_raw = cap.read()
        if not ret:
            print("\n❌ Error: Failed to read frame!")
            break

        collected_frame_count += 1
        # 更新进度提示
        progress_text = f"\r🔄 Total frames: {collected_frame_count}/{MAX_TOTAL_FRAMES} | Valid frames: {valid_frame_count}/{TARGET_VALID_FRAMES}"
        print(progress_text, end="")

        # 3. 核心：按照camera_undist.py的去畸变流程处理
        # 执行去畸变（使用映射表）
        undist = cv2.remap(frame_raw, mapx, mapy, cv2.INTER_LINEAR)
        
        # 裁剪并缩放以对齐（若alpha==0且roi有效）
        x, y, rw, rh = roi
        if rw > 0 and rh > 0 and ALPHA == 0:
            undist_crop = undist[y:y+rh, x:x+rw]
            try:
                undistorted_frame = cv2.resize(undist_crop, (w_raw, h_raw))
            except Exception:
                undistorted_frame = undist
        else:
            undistorted_frame = undist

        # 4. 预处理
        gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

        # 5. 检测ArUco标签（需要至少2个）
        corners, ids, rejected = aruco_detector.detectMarkers(gray_frame)
        sorted_x = None
        
        # 仅当检测到至少2个ArUco码时处理
        if ids is not None and len(ids) >= 2:
            # 取前两个ArUco码的角点
            two_aruco_corners = corners[:2]
            
            # 提取并排序两个码的第二/第三个角点（共4个）
            sorted_x = extract_and_sort_aruco_corners(two_aruco_corners, undistorted_frame)
            
            # 绘制两个ArUco码的基础轮廓和位姿（可视化）
            for i in range(2):  # 仅绘制前两个ArUco码
                tag_id = ids[i][0]
                tag_corners = corners[i][0]
                aruco.drawDetectedMarkers(undistorted_frame, [corners[i]], np.array([[tag_id]]), (0,0,255))
                # 绘制位姿轴
                rvec, tvec = calculate_aruco_pose(aruco_3d_points, tag_corners, newcameramtx, CAMERA_DIST)
                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(undistorted_frame, newcameramtx, CAMERA_DIST, rvec, tvec, TAG_SIZE/2, 2)

        # 6. 检测紫色小球
        ball_center = detect_purple_ball(
            undistorted_frame, PURPLE_HSV_LOW, PURPLE_HSV_HIGH,
            BALL_RADIUS_MIN, BALL_RADIUS_MAX,
            HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2
        )

        # 7. 筛选有效帧：同时满足 4个角点排序成功 + 小球检测成功
        if sorted_x is not None and len(sorted_x) == 4 and ball_center is not None:
            valid_frame_count += 1
            # 存储1-4点X坐标（排序后的4个角点）
            valid_frame_data[1].append(sorted_x[0])
            valid_frame_data[2].append(sorted_x[1])
            valid_frame_data[3].append(sorted_x[2])
            valid_frame_data[4].append(sorted_x[3])
            # 存储5点X坐标（小球中心）
            valid_frame_data[5].append(ball_center[0])

            # 绘制小球编号5（可视化）
            cv2.putText(undistorted_frame, "5", (ball_center[0]+8, ball_center[1]+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 显示画面（更新提示文本）
        cv2.putText(undistorted_frame, 
                    f"Total: {collected_frame_count}/{MAX_TOTAL_FRAMES} | Valid: {valid_frame_count}/{TARGET_VALID_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(undistorted_frame, 
                    "Need: 2 ArUco + 1 purple ball",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Collection Window", undistorted_frame)

        # 按键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🔴 Early stop collection!")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # ===================== 数据计算 =====================
    print(f"\n\n📊 Collection finished! Total frames: {collected_frame_count}, Valid frames: {valid_frame_count}")
    
    # 检查是否收集到足够的有效帧
    if valid_frame_count < TARGET_VALID_FRAMES:
        print(f"❌ Failed to collect {TARGET_VALID_FRAMES} valid frames (only got {valid_frame_count})!")
        return
    else:
        print(f"✅ Successfully collected {TARGET_VALID_FRAMES} valid frames!")

    # 1. 计算各点X坐标平均值（图像平面）
    avg_x = {}
    for point_id in [1,2,3,4,5]:
        avg_x[point_id] = np.mean(valid_frame_data[point_id])
        print(f"📈 Average X (image plane) of point {point_id}: {avg_x[point_id]:.2f} (from {len(valid_frame_data[point_id])} frames)")

    # 2. 打印点1-4的真实世界坐标
    print("\n🌍 Real-world coordinates of points 1-4:")
    for point_id in [1,2,3,4]:
        x, y, z = REAL_WORLD_COORDS[point_id]
        print(f"   Point {point_id}: x={x:.2f}, y={y:.2f}, z={z:.2f} mm")

    # 3. 计算图像平面差值项
    dx21 = avg_x[2] - avg_x[1]  # x2_img - x1_img
    dx31 = avg_x[3] - avg_x[1]  # x3_img - x1_img
    dx41 = avg_x[4] - avg_x[1]  # x4_img - x1_img
    dx51 = avg_x[5] - avg_x[1]  # x5_img - x1_img

    print(f"\n🔢 Image plane difference values:")
    print(f"x2-x1 = {dx21:.2f}, x3-x1 = {dx31:.2f}, x4-x1 = {dx41:.2f}, x5-x1 = {dx51:.2f}")

    # 4. 求解线性系数 a,b,c (x5-x1 = a*(x2-x1) + b*(x3-x1) + c*(x4-x1))
    # 按要求：令 a=0, b=0 求解c
    print("\n=====================================")
    print("🧮 Solve coefficients (x5-x1 = a*(x2-x1) + b*(x3-x1) + c*(x4-x1))")
    print("   Constraint: a=0, b=0")
    
    # 核心计算：令a=0、b=0，此时方程简化为 dx51 = c*dx41 → c = dx51/dx41
    if dx41 == 0:
        c = 0.0
        print(f"\n⚠️ Warning: x4-x1 = 0 (division by zero), set c=0")
    else:
        c = dx51 / dx41
    a = 0.0  # 强制设为0
    b = 0.0  # 强制设为0

    # 输出系数结果
    print(f"\n🔹 Coefficient Result:")
    print(f"   a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    
    # 验证图像平面计算
    calculated_dx51 = a * dx21 + b * dx31 + c * dx41
    print(f"\n🔍 Image plane verification:")
    print(f"   Calculated x5-x1 = {a:.4f}*({dx21:.2f}) + {b:.4f}*({dx31:.2f}) + {c:.4f}*({dx41:.2f}) = {calculated_dx51:.2f}")
    print(f"   Actual x5-x1 = {dx51:.2f}")
    print(f"   Error = {abs(calculated_dx51 - dx51):.2f} (absolute value)")

    # 5. 插值计算点5的真实世界坐标
    print("\n=====================================")
    print("🌐 Calculate point 5 real-world coordinates (interpolation):")
    # 提取点1和点4的真实坐标
    x1_real, y1_real, z1_real = REAL_WORLD_COORDS[1]
    x4_real, y4_real, z4_real = REAL_WORLD_COORDS[4]
    
    # 基于系数c插值计算点5的真实坐标
    # X轴：x5_real - x1_real = c * (x4_real - x1_real)
    x5_real = x1_real + c * (x4_real - x1_real)
    # Y轴：所有参考点Y坐标均为17.40，保持不变
    y5_real = y1_real
    # Z轴：z5_real - z1_real = c * (z4_real - z1_real)
    z5_real = z1_real + c * (z4_real - z1_real)
    
    # 整合点5真实坐标
    point5_real_coords = np.array([x5_real, y5_real, z5_real])
    
    # 输出点5真实坐标
    print(f"\n🔹 Point 5 Real-World Coordinates:")
    print(f"   x = {x5_real:.4f} mm")
    print(f"   y = {y5_real:.4f} mm")
    print(f"   z = {z5_real:.4f} mm")

    # 6. 保存点5真实坐标为NPY格式
    np.save(NPY_SAVE_PATH, point5_real_coords)
    print(f"\n💾 Point 5 coordinates saved to: {os.path.abspath(NPY_SAVE_PATH)}")
    print(f"   Saved data: {point5_real_coords}")
    print("=====================================")

# ===================== Run Entry =====================
if __name__ == "__main__":
    collect_frames_and_calculate()