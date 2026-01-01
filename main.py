import numpy as np
import json
import os
import math
from utils.ik import InverseKinematicsSolver
from utils.TrajectoryPlanning import Trajectory
from utils.SlaveTrajectoryFourier import MixedBasisTrajectoryOptimizer

# JAKA Mini 2 DH 参数 (参考 ik.py)
JAKA_DH = [
    [0,       np.pi/2,  0.187, 0],
    [0.210,   0,        0,     np.pi/2],
    [0,  np.pi/2,        0,     np.pi/2],
    [0,       np.pi/2,  0.2105, np.pi],
    [0,      np.pi/2,  0.006, np.pi],
    [0,       0,        0.1593, 0]
]

# JAKA Mini 2 关节限制 (弧度)
JAKA_JOINT_MIN = np.array([-2*np.pi, -2*np.pi/3, -2*np.pi/3, -2*np.pi, -2*np.pi/3, -2*np.pi])
JAKA_JOINT_MAX = np.array([ 2*np.pi,  2*np.pi/3,  2*np.pi/3,  2*np.pi,  2*np.pi/3,  2*np.pi])

def get_ik_solution(ik_solver, mode, value, ref_joints=None):
    """
    辅助函数：解析输入并获取关节角 (弧度)
    :param mode: 'joint' (角度制) 或 'pose' (位置+旋转矩阵)
    :param value: 对应的数值
    :param ik_solver: IK求解器实例
    :param ref_joints: 参考关节角(弧度)，用于IK选解
    """
    if mode == 'joint':
        # 输入为角度，转换为弧度
        return np.array([math.radians(x) for x in value])
    elif mode == 'pose':
        # 输入为位姿 (pos, rot)
        pos, rot = value
        # 如果没有参考关节角，使用零位
        if ref_joints is None:
            ref_joints = np.zeros(6)
        success, q, err = ik_solver.solve(pos, rot, initial_joints=ref_joints)
        if not success:
            raise RuntimeError(f"逆运动学求解失败: Pos={pos}, Err={err}")
        return q
    else:
        raise ValueError(f"未知模式: {mode}")

def generate_dual_arm_trajectory(master_waypoints, slave_waypoints, duration_segment=5.0):
    """
    生成双臂协同轨迹 (支持多个中间点)
    :param master_waypoints: 主臂路径点列表 [(mode, value), ...]
    :param slave_waypoints: 从臂路径点列表 [(mode, value), ...]
    :param duration_segment: 每个任务段的时间
    """
    print("="*50)
    print(f"开始双臂协同轨迹规划 (路径点数: {len(master_waypoints)})")
    print("="*50)

    if len(master_waypoints) != len(slave_waypoints):
        print("错误: 主从臂路径点数量不一致")
        return
    
    if len(master_waypoints) < 2:
        print("错误: 至少需要起始点和结束点")
        return

    freq = 125
    dt = 1.0 / freq
    
    master_filename = "master_trajectory_full.json"
    slave_filename = "slave_trajectory_full.json"
    
    # 临时文件模板
    temp_master_tpl = "temp_master_seg_{}.json"
    temp_slave_tpl = "temp_slave_seg_{}.json"

    # 基座参数
    Tbase_master = np.array([[1.0,0.0,0.0,0.275],[0.0,1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    Tbase_slave = np.array([[-1.0,0.0,0.0,-0.275],[0.0,-1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])

    # --- 1. 准备关键点 ---
    ik_solver = InverseKinematicsSolver(JAKA_DH, max_iter=2000, joint_limits=(JAKA_JOINT_MIN, JAKA_JOINT_MAX))
    
    master_qs = []
    slave_qs = []
    
    q_ref_m = np.zeros(6)
    q_ref_s = np.zeros(6)

    try:
        print("计算主臂关键点逆解...")
        for i, wp in enumerate(master_waypoints):
            q = get_ik_solution(ik_solver, wp[0], wp[1], ref_joints=q_ref_m)
            master_qs.append(q)
            q_ref_m = q # 更新参考点为当前点，防止突变
            print(f"  主臂点 {i}: {np.round(q, 3)}")

        print("计算从臂关键点逆解...")
        for i, wp in enumerate(slave_waypoints):
            q = get_ik_solution(ik_solver, wp[0], wp[1], ref_joints=q_ref_s)
            slave_qs.append(q)
            q_ref_s = q
            print(f"  从臂点 {i}: {np.round(q, 3)}")
            
    except Exception as e:
        print(f"关键点计算错误: {e}")
        return

    # 辅助函数：生成主臂轨迹并保存
    def generate_and_save_master(q_s, q_e, T, filename):
        traj = Trajectory(T, q_s, q_e)
        traj.tra_plan()
        data = []
        steps = int(T * freq)
        for i in range(steps + 1):
            t = i * dt
            if t > T: t = T
            q, _ = traj.evaluate(t)
            data.append({"Joint": [float(x) for x in q]})
        with open(filename, 'w') as f:
            json.dump(data, f)
        return data

    # 辅助函数：优化从臂轨迹
    def optimize_slave(q_s, q_e, T, master_file, output_file):
        planner = MixedBasisTrajectoryOptimizer(
            T=T,
            q_s=q_s,
            q_e=q_e,
            Tbase_master=Tbase_master,
            Tbase_slave=Tbase_slave,
            master_trajectory_file=master_file,
            d_min=0.2,
            d_max=0.6,
            dt=dt,
            use_fourier=True,
            n_harmonics=2
        )
        if planner.optimize(max_iter=500, disp=True):
            planner.save_trajectory(output_file)
            with open(output_file, 'r') as f:
                return json.load(f)
        return None

    # --- 2. 生成轨迹 ---
    
    full_master_data = []
    full_slave_data = []
    
    # 任务阶段 (Point i -> Point i+1)
    num_segments = len(master_qs) - 1
    
    for i in range(num_segments):
        print(f"\n[阶段 {i+1}] 生成任务段 {i+1}/{num_segments} (Point {i} -> Point {i+1}, {duration_segment}s)...")
        
        # 如果不是第一段，去除上一段的最后一个点，避免重复 (因为上一段的终点等于这一段的起点)
        if full_master_data:
            full_master_data.pop()
            full_slave_data.pop()
        
        tm_file = temp_master_tpl.format(i)
        ts_file = temp_slave_tpl.format(i)
        
        seg_m = generate_and_save_master(master_qs[i], master_qs[i+1], duration_segment, tm_file)
        seg_s = optimize_slave(slave_qs[i], slave_qs[i+1], duration_segment, tm_file, ts_file)
        
        if seg_s is None: 
            print(f"错误: 任务段 {i+1} 优化失败")
            return
            
        full_master_data.extend(seg_m)
        full_slave_data.extend(seg_s)
        
        if os.path.exists(tm_file): os.remove(tm_file)
        if os.path.exists(ts_file): os.remove(ts_file)

    # --- 3. 保存 ---
    print("\n保存完整轨迹...")
    with open(master_filename, 'w') as f:
        json.dump(full_master_data, f, indent=2)
    with open(slave_filename, 'w') as f:
        json.dump(full_slave_data, f, indent=2)

    print(f"双臂完整轨迹生成完成!")
    print(f"主臂轨迹: {os.path.abspath(master_filename)}")
    print(f"从臂轨迹: {os.path.abspath(slave_filename)}")

if __name__ == "__main__":
    # 定义主臂和从臂的路径点列表
    # 模式支持: 'joint' (角度), 'pose' (位姿)

    # --- 主臂路径点 ---
    master_wps = [
        ('joint', [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]), # Start
        ('joint', [-46.100, -10.193, 97.777, -4.253, 68.200, -37.112]), # Mid
        ('joint', [-60.572, 13.115, 113.291, 2.180, 48.550, -57.954]),  # Pick
        ('joint', [-31.551, 29.221, 31.079, -78.389, 59.712, -57.965])  # End
        # 可以继续添加更多点...
    ]
    
    # --- 从臂路径点 ---
    slave_wps = [
        ('joint', [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]), # Start
        ('joint', [60.903, -23.849, 96.589, 1.014, 89.364, -11.038]),   # Mid
        ('joint', [59.285, 17.656, 104.646, -3.474, 63.511, -17.628]),  # Pick
        ('joint', [31.659, 27.489, 96.695, -74.691, -65.946, -17.627])  # End
        # 数量必须与主臂一致
    ]

    generate_dual_arm_trajectory(master_wps, slave_wps, duration_segment=5.0)
