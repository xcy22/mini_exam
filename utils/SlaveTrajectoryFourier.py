from utils import SlaveTrajectory3
import numpy as np
import json
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import math

class MixedBasisTrajectoryOptimizer(SlaveTrajectory3.SlaveTrajectoryOptimizer):
    """
    混合基函数轨迹优化器
    使用多项式+正弦函数的混合基函数
    """
    
    def __init__(self, *args, use_fourier=False, n_harmonics=2, **kwargs):
        """
        初始化混合基函数轨迹优化器
        
        参数:
        use_fourier: 是否使用傅里叶级数项
        n_harmonics: 傅里叶级数的谐波数量
        """
        super().__init__(*args, **kwargs)
        
        # 混合基函数设置
        self.use_fourier = use_fourier
        self.n_harmonics = n_harmonics
        
        # 调整系数数量
        if use_fourier:
            # 基函数: 1 (常数) + t + t² + t³ + sin(ωt) + cos(ωt) + ...
            # 对于每个谐波：一个sin项，一个cos项
            self.n_fourier_coeffs = 2 * n_harmonics  # sin和cos系数
            self.n_poly_coeffs = 4  # 多项式项数 (1, t, t², t³)
            self.n_coeffs_per_joint = self.n_poly_coeffs + self.n_fourier_coeffs
        else:
            # 使用5次多项式
            self.n_coeffs_per_joint = 6
        
        self.n_total_coeffs = self.n_joints * self.n_coeffs_per_joint
        
        # 基函数频率
        self.omega = 2 * np.pi / self.T  # 基频
    
    # 重写所有使用系数直接解包的约束函数
    
    def _eq_constraint_end_position(self, coeffs):
        """等式约束：终点位置 - 适用于任意基函数"""
        constraints = np.zeros(self.n_joints, dtype=np.float64)
        
        # 使用compute_position方法计算终点位置
        q_end = self.compute_position(coeffs, self.T)
        
        for i in range(self.n_joints):
            constraints[i] = q_end[i] - self.q_e[i]
        
        return constraints
    
    def _eq_constraint_end_velocity(self, coeffs):
        """等式约束：终点速度 - 适用于任意基函数"""
        constraints = np.zeros(self.n_joints, dtype=np.float64)
        
        # 使用compute_velocity方法计算终点速度
        v_end = self.compute_velocity(coeffs, self.T)
        
        for i in range(self.n_joints):
            constraints[i] = v_end[i] - 0.0  # v(T) = 0
        
        return constraints
    
    # 重写其他可能直接解包系数的方法
    
    def _coeffs_to_matrix(self, coeffs):
        """将一维系数数组转换为矩阵形式"""
        return coeffs.reshape(self.n_joints, self.n_coeffs_per_joint)
    
    def _get_initial_guess(self):
        """生成混合基函数的初始猜测"""
        coeffs = np.zeros(self.n_total_coeffs, dtype=np.float64)
        
        for i in range(self.n_joints):
            q0 = self.q_s[i]
            qf = self.q_e[i]
            T = self.T
            
            if self.use_fourier:
                # 多项式部分（与原始三次多项式相同）
                a0 = q0
                a1 = 0.0
                a2 = (3.0 * (qf - q0)) / (T**2)
                a3 = (-2.0 * (qf - q0)) / (T**3)
                
                # 傅里叶部分初始为0
                fourier_coeffs = [0.0] * (2 * self.n_harmonics)
                
                # 合并系数
                joint_coeffs = [a0, a1, a2, a3] + fourier_coeffs
            else:
                # 五次多项式
                a0 = q0
                a1 = 0.0
                a2 = (3.0 * (qf - q0)) / (T**2)
                a3 = (-2.0 * (qf - q0)) / (T**3)
                a4 = 0.0
                a5 = 0.0
                joint_coeffs = [a0, a1, a2, a3, a4, a5]
            
            # 存储系数
            idx = i * self.n_coeffs_per_joint
            coeffs[idx:idx+self.n_coeffs_per_joint] = joint_coeffs
        
        return coeffs
    
    def compute_position(self, coeffs, t):
        """计算位置（混合基函数）"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        positions = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            if self.use_fourier:
                # 提取系数
                poly_coeffs = coeffs_matrix[i, :4]
                fourier_coeffs = coeffs_matrix[i, 4:]
                
                # 多项式部分
                poly_term = poly_coeffs[0] + poly_coeffs[1]*t + poly_coeffs[2]*(t**2) + poly_coeffs[3]*(t**3)
                
                # 傅里叶部分
                fourier_term = 0.0
                for h in range(self.n_harmonics):
                    sin_coeff = fourier_coeffs[2*h]
                    cos_coeff = fourier_coeffs[2*h + 1]
                    omega_h = (h + 1) * self.omega
                    fourier_term += sin_coeff * np.sin(omega_h * t) + cos_coeff * np.cos(omega_h * t)
                
                positions[i] = poly_term + fourier_term
            else:
                # 五次多项式
                a0, a1, a2, a3, a4, a5 = coeffs_matrix[i]
                positions[i] = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4) + a5*(t**5)
        
        return positions
    
    def compute_velocity(self, coeffs, t):
        """计算速度（混合基函数）"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        velocities = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            if self.use_fourier:
                # 提取系数
                poly_coeffs = coeffs_matrix[i, :4]
                fourier_coeffs = coeffs_matrix[i, 4:]
                
                # 多项式部分速度
                poly_vel = poly_coeffs[1] + 2*poly_coeffs[2]*t + 3*poly_coeffs[3]*(t**2)
                
                # 傅里叶部分速度
                fourier_vel = 0.0
                for h in range(self.n_harmonics):
                    sin_coeff = fourier_coeffs[2*h]
                    cos_coeff = fourier_coeffs[2*h + 1]
                    omega_h = (h + 1) * self.omega
                    fourier_vel += sin_coeff * omega_h * np.cos(omega_h * t) - cos_coeff * omega_h * np.sin(omega_h * t)
                
                velocities[i] = poly_vel + fourier_vel
            else:
                # 五次多项式速度
                a0, a1, a2, a3, a4, a5 = coeffs_matrix[i]
                velocities[i] = a1 + 2*a2*t + 3*a3*(t**2) + 4*a4*(t**3) + 5*a5*(t**4)
        
        return velocities
    
    def compute_acceleration(self, coeffs, t):
        """计算加速度（混合基函数）"""
        coeffs_matrix = self._coeffs_to_matrix(coeffs)
        accelerations = np.zeros(self.n_joints, dtype=np.float64)
        
        for i in range(self.n_joints):
            if self.use_fourier:
                # 提取系数
                poly_coeffs = coeffs_matrix[i, :4]
                fourier_coeffs = coeffs_matrix[i, 4:]
                
                # 多项式部分加速度
                poly_acc = 2*poly_coeffs[2] + 6*poly_coeffs[3]*t
                
                # 傅里叶部分加速度
                fourier_acc = 0.0
                for h in range(self.n_harmonics):
                    sin_coeff = fourier_coeffs[2*h]
                    cos_coeff = fourier_coeffs[2*h + 1]
                    omega_h = (h + 1) * self.omega
                    fourier_acc += -sin_coeff * (omega_h**2) * np.sin(omega_h * t) - cos_coeff * (omega_h**2) * np.cos(omega_h * t)
                
                accelerations[i] = poly_acc + fourier_acc
            else:
                # 五次多项式加速度
                a0, a1, a2, a3, a4, a5 = coeffs_matrix[i]
                accelerations[i] = 2*a2 + 6*a3*t + 12*a4*(t**2) + 20*a5*(t**3)
        
        return accelerations
    
    # 可选：也重写起点位置和速度约束以确保一致性
    def _eq_constraint_start_position(self, coeffs):
        """等式约束：起点位置 - 使用compute_position确保一致性"""
        constraints = np.zeros(self.n_joints, dtype=np.float64)
        
        # 使用compute_position方法计算起点位置
        q_start = self.compute_position(coeffs, 0.0)
        
        for i in range(self.n_joints):
            constraints[i] = q_start[i] - self.q_s[i]
        
        return constraints
    
    def _eq_constraint_start_velocity(self, coeffs):
        """等式约束：起点速度 - 使用compute_velocity确保一致性"""
        constraints = np.zeros(self.n_joints, dtype=np.float64)
        
        # 使用compute_velocity方法计算起点速度
        v_start = self.compute_velocity(coeffs, 0.0)
        
        for i in range(self.n_joints):
            constraints[i] = v_start[i] - 0.0  # v(0) = 0
        
        return constraints
    
    def validate_constraints(self, tolerance=1e-4):
        """
        重写验证约束方法，适应混合基函数
        
        参数:
        tolerance: 允许的误差范围
        
        返回:
        all_satisfied: 是否所有约束都满足
        """
        if self.optimized_coeffs is None:
            raise ValueError("请先执行optimize()方法进行优化")
        
        print("\n验证约束条件:")
        
        all_satisfied = True
        
        # 验证等式约束
        print("1. 等式约束验证:")
        
        # 使用子类方法计算起点位置和速度
        q_start = self.compute_position(self.optimized_coeffs, 0.0)
        v_start = self.compute_velocity(self.optimized_coeffs, 0.0)
        
        # 起点位置
        for i in range(self.n_joints):
            error = abs(q_start[i] - self.q_s[i])
            if error > tolerance:
                print(f"   关节{i+1}: 起点位置误差 {error:.6f} > {tolerance}")
                all_satisfied = False
        
        # 终点位置
        q_end = self.compute_position(self.optimized_coeffs, self.T)
        for i in range(self.n_joints):
            error = abs(q_end[i] - self.q_e[i])
            if error > tolerance:
                print(f"   关节{i+1}: 终点位置误差 {error:.6f} > {tolerance}")
                all_satisfied = False
        
        # 起点速度
        for i in range(self.n_joints):
            error = abs(v_start[i] - 0.0)
            if error > tolerance:
                print(f"   关节{i+1}: 起点速度误差 {error:.6f} > {tolerance}")
                all_satisfied = False
        
        # 终点速度
        v_end = self.compute_velocity(self.optimized_coeffs, self.T)
        for i in range(self.n_joints):
            error = abs(v_end[i] - 0.0)
            if error > tolerance:
                print(f"   关节{i+1}: 终点速度误差 {error:.6f} > {tolerance}")
                all_satisfied = False
        
        # 验证不等式约束
        print("\n2. 不等式约束验证:")
        
        # 采样检查关节位置约束
        n_check = 10
        time_check = np.linspace(0, self.T, n_check)
        
        for t in time_check:
            positions = self.compute_position(self.optimized_coeffs, t)
            velocities = self.compute_velocity(self.optimized_coeffs, t)
            
            for i in range(self.n_joints):
                # 检查位置约束
                if positions[i] < self.q_min[i] - tolerance:
                    print(f"   时间{t:.2f}s, 关节{i+1}: 位置 {positions[i]:.4f} < 下限 {self.q_min[i]:.4f}")
                    all_satisfied = False
                elif positions[i] > self.q_max[i] + tolerance:
                    print(f"   时间{t:.2f}s, 关节{i+1}: 位置 {positions[i]:.4f} > 上限 {self.q_max[i]:.4f}")
                    all_satisfied = False
                
                # 检查速度约束
                if velocities[i] < self.v_min[i] - tolerance:
                    print(f"   时间{t:.2f}s, 关节{i+1}: 速度 {velocities[i]:.4f} < 下限 {self.v_min[i]:.4f}")
                    all_satisfied = False
                elif velocities[i] > self.v_max[i] + tolerance:
                    print(f"   时间{t:.2f}s, 关节{i+1}: 速度 {velocities[i]:.4f} > 上限 {self.v_max[i]:.4f}")
                    all_satisfied = False
        
        # 验证末端距离约束
        print("\n3. 末端距离约束验证:")
        n_check = 10
        time_indices = np.linspace(0, self.n_points-1, n_check, dtype=int)
        
        for idx in time_indices:
            t = idx * self.dt
            q_slave = self.compute_position(self.optimized_coeffs, t)
            slave_pos = self.forward_kinematics(q_slave, self.Tbase_slave)
            
            if idx < len(self.master_positions):
                master_pos = self.master_positions[idx]
            else:
                master_pos = self.master_positions[-1]
            
            distance = np.linalg.norm(slave_pos - master_pos)
            
            if distance < self.d_min - tolerance:
                print(f"   时间{t:.2f}s: 距离 {distance:.4f} < 下限 {self.d_min:.4f}")
                all_satisfied = False
            elif distance > self.d_max + tolerance:
                print(f"   时间{t:.2f}s: 距离 {distance:.4f} > 上限 {self.d_max:.4f}")
                all_satisfied = False
            else:
                print(f"   时间{t:.2f}s: 距离 {distance:.4f} 在范围内 [{self.d_min:.4f}, {self.d_max:.4f}]")
        
        if all_satisfied:
            print("\n所有约束条件满足!")
        else:
            print("\n警告: 部分约束条件不满足!")
        
        return all_satisfied


def main():
    """主函数示例"""
    print("从臂轨迹规划示例")
    print("=" * 50)
    
    # 从臂轨迹参数
    T = 200.0  # 总时间5秒
    dt = 1/125  # 时间步长
    
    # 从臂起始和结束位置（6个关节）
    qs_d=[59.285,17.656,104.646,-3.474,63.511,-17.628]
    qe_d=[31.659,27.489,96.695,-74.691,-65.946,-17.627]
    qs_r=[0.0,0.0,0.0,0.0,0.0,0.0]
    qe_r=[0.0,0.0,0.0,0.0,0.0,0.0]
    
    for i in range(len(qs_d)):
        qs_r[i]=math.radians(qs_d[i])
        qe_r[i]=math.radians(qe_d[i])
    
    # 创建从臂轨迹规划器
    print(f"创建从臂轨迹规划器...")
    print(f"起始位置: {qs_r}")
    print(f"结束位置: {qe_r}")
    
    planner = MixedBasisTrajectoryOptimizer(
        T=T,
        q_s=qs_r,
        q_e=qe_r,
        Tbase_master=np.array([[1.0,0.0,0.0,0.275],[0.0,1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]),
        Tbase_slave=np.array([[-1.0,0.0,0.0,-0.275],[0.0,-1.0,0.0,-0.2],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]),
        master_trajectory_file='trajectory_data.json',
        d_min=0.25,   # 最小距离20cm
        d_max=0.32,   # 最大距离30cm
        dt=dt,
        use_fourier=True,
        n_harmonics=2
    )
    

    start_dist, end_dist = planner.diagnose_optimization()
    # 执行优化
    success = planner.optimize(
        max_iter=1000,      # 增加迭代次数
        disp=True           # 显示详细信息
    )
    
    if success:
        # 验证约束
        planner.validate_constraints(tolerance=1e-3)
        
        # 保存轨迹
        planner.save_trajectory('slave_trajectory_data.json')
        
        # 绘制轨迹
        try:
            planner.plot_trajectory()
        except:
            print("绘图功能需要matplotlib库，请安装: pip install matplotlib")
    else:
        print("优化失败，无法生成有效轨迹")


if __name__ == "__main__":
    main()