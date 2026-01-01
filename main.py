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

def generate_dual_arm_trajectory(master_start, master_mid, master_end, slave_start, slave_mid, slave_end):
    """
    生成双臂协同轨迹 (含中间点)
    参数格式: (mode, value)
    """
    print("="*50)
    print("开始双臂协同轨迹规划 (含归位阶段 + 中间点)")
    print("="*50)

    # --- 参数设置 ---
    duration_approach = 5.0 # 阶段1：归位时间
    duration_task1 = 5.0    # 阶段2：任务第一段 (Start -> Mid)
    duration_task2 = 5.0    # 阶段3：任务第二段 (Mid -> End)
    freq = 125
    dt = 1.0 / freq
    
    master_filename = "master_trajectory_full.json"
    slave_filename = "slave_trajectory_full.json"
    
    # 临时文件
    temp_master_p1 = "temp_master_p1.json"
    temp_slave_p1 = "temp_slave_p1.json"
    temp_master_p2 = "temp_master_p2.json"
    temp_slave_p2 = "temp_slave_p2.json"
    temp_master_p3 = "temp_master_p3.json"
    temp_slave_p3 = "temp_slave_p3.json"

    # 基座参数
    Tbase_master = np.array([[1.0,0.0,0.0,0.275],[0.0,1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    Tbase_slave = np.array([[-1.0,0.0,0.0,-0.275],[0.0,-1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])

    # --- 1. 准备关键点 ---
    
    # 初始化 IK
    ik_solver = InverseKinematicsSolver(JAKA_DH, max_iter=2000, joint_limits=(JAKA_JOINT_MIN, JAKA_JOINT_MAX))
    q_zero = np.zeros(6)

    try:
        print("计算主臂关键点...")
        q_start_m = get_ik_solution(ik_solver, master_start[0], master_start[1], ref_joints=q_zero)
        q_mid_m = get_ik_solution(ik_solver, master_mid[0], master_mid[1], ref_joints=q_start_m)
        q_end_m = get_ik_solution(ik_solver, master_end[0], master_end[1], ref_joints=q_mid_m)

        print("计算从臂关键点...")
        q_start_s = get_ik_solution(ik_solver, slave_start[0], slave_start[1], ref_joints=q_zero)
        q_mid_s = get_ik_solution(ik_solver, slave_mid[0], slave_mid[1], ref_joints=q_start_s)
        q_end_s = get_ik_solution(ik_solver, slave_end[0], slave_end[1], ref_joints=q_mid_s)
    except Exception as e:
        print(f"关键点计算错误: {e}")
        return

    print(f"主臂起始关节(rad): {np.round(q_start_m, 3)}")
    print(f"主臂中间关节(rad): {np.round(q_mid_m, 3)}")
    print(f"从臂起始关节(rad): {np.round(q_start_s, 3)}")
    print(f"从臂中间关节(rad): {np.round(q_mid_s, 3)}")

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

    # --- 2. 生成阶段 1 (归位) 数据 ---
    print(f"\n[阶段 1] 生成归位轨迹 (Zero -> Start, {duration_approach}s)...")
    
    # 2.1 主臂归位 (多项式)
    master_phase1 = generate_and_save_master(q_zero, q_start_m, duration_approach, temp_master_p1)
    
    # 2.2 从臂归位 (优化)
    slave_phase1 = optimize_slave(q_zero, q_start_s, duration_approach, temp_master_p1, temp_slave_p1)
    if slave_phase1 is None:
        print("错误: 阶段1从臂优化失败")
        return

    # --- 3. 生成阶段 2 (任务第一段) 数据 ---
    print(f"\n[阶段 2] 生成任务第一段轨迹 (Start -> Mid, {duration_task1}s)...")

    # 3.1 主臂任务1
    master_phase2 = generate_and_save_master(q_start_m, q_mid_m, duration_task1, temp_master_p2)

    # 3.2 从臂任务1
    slave_phase2 = optimize_slave(q_start_s, q_mid_s, duration_task1, temp_master_p2, temp_slave_p2)
    if slave_phase2 is None:
        print("错误: 阶段2从臂优化失败")
        return

    # --- 4. 生成阶段 3 (任务第二段) 数据 ---
    print(f"\n[阶段 3] 生成任务第二段轨迹 (Mid -> End, {duration_task2}s)...")

    # 4.1 主臂任务2
    master_phase3 = generate_and_save_master(q_mid_m, q_end_m, duration_task2, temp_master_p3)

    # 4.2 从臂任务2
    slave_phase3 = optimize_slave(q_mid_s, q_end_s, duration_task2, temp_master_p3, temp_slave_p3)
    if slave_phase3 is None:
        print("错误: 阶段3从臂优化失败")
        return

    # --- 5. 合并与保存 ---
    print("\n合并轨迹数据...")
    
    # 拼接列表 (去除前一段的最后一个点，因为它与下一段的第一个点重合)
    master_full = master_phase1[:-1] + master_phase2[:-1] + master_phase3
    slave_full = slave_phase1[:-1] + slave_phase2[:-1] + slave_phase3
    
    # 保存最终文件
    with open(master_filename, 'w') as f:
        json.dump(master_full, f, indent=2)
        
    with open(slave_filename, 'w') as f:
        json.dump(slave_full, f, indent=2)

    # 清理临时文件
    for f in [temp_master_p1, temp_slave_p1, temp_master_p2, temp_slave_p2, temp_master_p3, temp_slave_p3]:
        if os.path.exists(f): os.remove(f)

    print(f"双臂完整轨迹生成完成!")
    print(f"主臂轨迹: {os.path.abspath(master_filename)}")
    print(f"从臂轨迹: {os.path.abspath(slave_filename)}")

if __name__ == "__main__":
    # 定义主臂和从臂的起始点、中间点与目标点
    # 模式支持: 
    #   'joint': 关节角度 (单位: 度), 格式: [j1, j2, j3, j4, j5, j6]
    #   'pose':  末端位姿, 格式: ([x, y, z], [[r11, r12, r13], ...])

    # --- 主臂配置 ---
    # 示例: 使用关节角
    master_start_conf = ('joint', [-46.100, -10.193, 97.777, -4.253, 68.200, -37.112])
    # 示例中间点 (请根据实际需求修改)
    master_mid_conf   = ('joint', [-60.572, 13.115, 113.291, 2.180, 48.550, -57.954]) 
    master_end_conf   = ('joint', [-31.551, 29.221, 31.079, -78.389, 59.712, -57.965])
    
    # --- 从臂配置 ---
    slave_start_conf = ('joint', [60.903, -23.849, 96.589, 1.014, 89.364, -11.038])
    # 示例中间点 (请根据实际需求修改)
    slave_mid_conf   = ('joint', [59.285, 17.656, 104.646, -3.474, 63.511, -17.628])
    slave_end_conf   = ('joint', [31.659, 27.489, 96.695, -74.691, -65.946, -17.627])

    generate_dual_arm_trajectory(master_start_conf, master_mid_conf, master_end_conf, 
                                 slave_start_conf, slave_mid_conf, slave_end_conf)
