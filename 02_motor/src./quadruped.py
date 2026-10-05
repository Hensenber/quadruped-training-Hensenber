import time
import numpy as np
import mujoco
import mujoco.viewer
from model_utils import print_model_info,get_joint_indices,get_actuator_id,get_motor_indices
from enum import Enum, auto

MODEL_PATH = "models/robot/black_description.xml"

#引入狗趴姿的一组参数
CROUCH_QPOS = np.array([
    # trunk position
    0.0113025555,
    0.00516957988,
    0.144941866,

    # trunk quaternion: w, x, y, z
    0.999937920,
    0.000000808559040,
    0.0000122068091,
    0.0111425210,

    # FL
     0.501394745,
     1.60080545,
    -2.55423629,

    # FR
    -0.501393610,
    -1.60080306,
     2.55423900,

    # RR
     0.501359362,
    -1.60081606,
     2.55422931,

    # RL
    -0.501359963,
     1.60081862,
    -2.55422686,
])

MOVE_DURATION = 2.0             #平滑运动时间
SIM_DURATION = 4.0              #总仿真时间,s
DISABLE_GRAVITY = False          #关闭/开启重力
DISABLE_CONTACT = False          #关闭/开启接触



#使用Enum来枚举状态机的状态:阻尼模式与站立模式
class RobotState(Enum):                 #枚举类型一般用驼峰写法PascalCase
    DAMPING = auto()
    STANDING = auto()

#定义状态机类
class RobotMachine:
    def __init__(self):
        self.state = RobotState.DAMPING #默认进入DAMPING模式
        self.start_time = None          #记录站立开始时间,DAMPING模式下为None
        self.start_angles = None         #记录站立开始角度,DAMPING模式下为None

    def enter_STANDING(self,current_time,current_angles):
        if self.state == RobotState.STANDING:
            return
        
        else:
            self.state = RobotState.STANDING
            self.start_time = current_time
            self.start_angles = current_angles.copy()

        print("[FSM] DAMPING->STANDING")

    def enter_DAMPING(self):
        if self.state == RobotState.DAMPING:
            return

        else:
            self.state = RobotState.DAMPING
            self.start_time = None
            self.start_angles = None

        print("[FSM] STANDING->DAMPING")

#使用mujoco自带的键盘处理函数来处理外部键盘输入
from queue import SimpleQueue,Empty
def handle_key(keycode,pending_commmands):
    #将按键转换成状态切换命令
    if keycode == ord("["):
        pending_commmands.put(RobotState.STANDING)
    elif keycode == ord("]"):
        pending_commmands.put(RobotState.DAMPING)


#核心参数设置
Kp = 80.0                       #虚拟弹簧刚度系数
Kd = 3.0                        #阻尼系数
Kd_DAMPING = 3.0                #阻尼模式阻尼系数

#修改initial_angles与target_angles中的参数即可控制狗的不同姿态
#calf控制小腿的摆动角
INITIAL_ANGLES = {
    "FL_hip_joint":   0.0,
    "FL_thigh_joint": 0.43,
    "FL_calf_joint": -0.85,

    "FR_hip_joint":   0.0,
    "FR_thigh_joint": -0.43,
    "FR_calf_joint":  0.85,

    "RR_hip_joint":   0.0,
    "RR_thigh_joint": -0.43,
    "RR_calf_joint":  0.85,

    "RL_hip_joint":   0.0,
    "RL_thigh_joint": 0.43,
    "RL_calf_joint": -0.85,
}

TARGET_ANGLES = {
    "FL_hip_joint":   0.0,
    "FL_thigh_joint": 0.43,
    "FL_calf_joint": -0.85,

    "FR_hip_joint":   0.0,
    "FR_thigh_joint": -0.43,
    "FR_calf_joint":  0.85,

    "RR_hip_joint":   0.0,
    "RR_thigh_joint": -0.43,
    "RR_calf_joint":  0.85,

    "RL_hip_joint":   0.0,
    "RL_thigh_joint": 0.43,
    "RL_calf_joint": -0.85,
}

def motor_control(q,dq,q_des,dq_des,tau_ff,Kp,Kd):
    tau = tau_ff + Kp * (q_des - q) + Kd * (dq_des - dq)
    return tau


#建立12个电机的索引映射
def build_motor_mapping(model):
    motors = []
    used_joints = set()

    if model.nu != 12:
        raise RuntimeError(
            f"期望12个电机，实际为 {model.nu} 个"
        )

    for actuator_id in range(model.nu):

        # 获取电机名称
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id
        )

        # 仅支持直接连接到关节的 motor
        if (
            model.actuator_trntype[actuator_id]
            != mujoco.mjtTrn.mjTRN_JOINT
        ):
            raise RuntimeError(
                f"{actuator_name} 不是直接关节驱动"
            )

        # 读取该电机连接的关节
        joint_id = int(
            model.actuator_trnid[actuator_id, 0]
        )

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id
        )

        if (
            model.jnt_type[joint_id]
            != mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise RuntimeError(
                f"{joint_name} 不是 hinge 关节"
            )

        if joint_id in used_joints:
            raise RuntimeError(
                f"{joint_name} 被多个电机驱动，"
                "当前程序不支持这种配置"
            )

        used_joints.add(joint_id)

        # 复用 model_utils.py
        qpos_id, qvel_id, ctrl_id = get_motor_indices(
            model,
            joint_name,
            actuator_name
        )

        # 检查电机连接关系
        if int(model.actuator_trnid[ctrl_id, 0]) != joint_id:
            raise RuntimeError(
                f"{actuator_name} 连接关系不正确"
            )

        # 当前控制器假设 ctrl 就是关节力矩
        if not np.isclose(
            model.actuator_gear[ctrl_id, 0], 1.0
        ):
            raise RuntimeError(
                f"{actuator_name} 的 gear 不是1，"
                "需要先处理传动比"
            )

        if not model.jnt_limited[joint_id]:
            raise RuntimeError(
                f"{joint_name} 没有设置关节限位"
            )

        q_min, q_max = model.jnt_range[joint_id]

        #从用户定义的角度字典读取初始姿态和目标姿态
        q_start = INITIAL_ANGLES[joint_name]
        q_target = TARGET_ANGLES[joint_name]

        if not (q_min <= q_start <= q_max):
            raise ValueError(
            f"{joint_name}: 初始角度 {q_start} 超出限位 [{q_min}, {q_max}]"
        )

        if not (q_min <= q_target <= q_max):
            raise ValueError(
            f"{joint_name}: 目标角度 {q_target} 超出限位 [{q_min}, {q_max}]"
        )


        motors.append({
            "joint_name": joint_name,
            "actuator_name": actuator_name,
            "qpos_id": qpos_id,
            "qvel_id": qvel_id,
            "ctrl_id": ctrl_id,
            "q_start": q_start,
            "q_target": q_target,
        })

    if len(used_joints) != 12:
        raise RuntimeError("没有找到12个独立的受控关节")

    return motors

def get_desired_state(t, q_start, q_target):
    s = np.clip(t / MOVE_DURATION, 0.0, 1.0)

    q_des = q_start + s * (q_target - q_start)

    if 0.0 <= t < MOVE_DURATION:
        dq_des = (q_target - q_start) / MOVE_DURATION
    else:
        dq_des = 0.0

    return q_des, dq_des



def main():
    #加载模型
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print_model_info(model)

    # 关闭重力
    if DISABLE_GRAVITY:
        model.opt.gravity[:] = 0.0

    # 关闭接触
    if DISABLE_CONTACT:
        model.opt.disableflags |= (
            mujoco.mjtDisableBit.mjDSBL_CONTACT
        )


    # 建立索引映射
    motors = build_motor_mapping(model)

    print("\n=== Motor Mapping ===")

    for motor in motors:
        print(
            f"{motor['actuator_name']} -> "
            f"{motor['joint_name']} | "
            f"start={motor['q_start']:.3f}, "
            f"target={motor['q_target']:.3f}"
        )
    
    # 设置初始关节位置
    if model.nq != len(CROUCH_QPOS):
        raise ValueError(
            f"初始姿态长度不匹配: "
            f"nq={model.nq}, "
            f"len(CROUCH_QPOS)={len(CROUCH_QPOS)}"
        )

    data.qpos[:] = CROUCH_QPOS
    data.qvel[:] = 0.0    #直接引入稳定趴姿
    

    """
    #设置机身初始位置与姿态
    data.qpos[:] = 0.0
    data.qpos[2] = 0.5
    data.qpos[3] = 1.0

    #根据INITIAL_ANGLES设置12个初始关节角度
    for motor in motors:
        data.qpos[motor["qpos_id"]] = motor["q_start"]
    
    data.qvel[:] = 0.0  #初始速度依然设置为0
    """

    mujoco.mj_forward(model, data)
    # 保存时间变量
    last_print_time = -0.5

    print("\n=== Start 12-Joint PD ===")

    #创建状态机
    fsm = RobotMachine()
    # 静态站立实验：启动时直接进入 STANDING
    current_angles = np.array([
        data.qpos[motor["qpos_id"]]
        for motor in motors
    ])

    fsm.enter_STANDING(
        current_time=data.time,
        current_angles=current_angles
    )
    #创建键盘命令队列
    pending_commands = SimpleQueue()

    # 定义Viewer键盘回调
    def on_key(keycode):
        handle_key(keycode,pending_commands)
    

    #启动可视化
    with mujoco.viewer.launch_passive(model,data,key_callback=on_key) as viewer:
        wall_start = time.perf_counter()
        RENDER_INTERVAL = 1.0 / 60.0
        last_render_time = data.time - RENDER_INTERVAL

        while viewer.is_running():
            #处理键盘命令
            while True:
                try:
                    command = pending_commands.get_nowait()
                except Empty:
                    break

                if command == RobotState.STANDING:
                    current_angles = np.array([data.qpos[motor["qpos_id"]]for motor in motors])
                    fsm.enter_STANDING(current_time=data.time,current_angles=current_angles)

                elif command == RobotState.DAMPING:
                    fsm.enter_DAMPING()

            # 统计当前仿真步的控制情况
            saturated_count = 0
            max_tracking_error = 0.0

            # 计算12个关节的力矩
            for i, motor in enumerate(motors):
                
                q = data.qpos[motor["qpos_id"]]
                dq = data.qvel[motor["qvel_id"]]

                if fsm.state == RobotState.DAMPING:
                    # 阻尼模式：不进行位置控制
                    tau = -Kd_DAMPING * dq

                elif fsm.state == RobotState.STANDING:
                    #计算本次动作的实际进行时间
                    elapsed_time = data.time - fsm.start_time
                    #获取本次动作的实际起始角度
                    q_start = fsm.start_angles[i]
                    #读取预设站立目标角度
                    q_target = motor["q_target"]
                    #生成平滑插值轨迹
                    q_des,dq_des = get_desired_state(elapsed_time,q_start,q_target)
                    error = abs(q_des - q)
                    max_tracking_error = max(max_tracking_error, error)
                    tau = motor_control(q=q,dq=dq,q_des=q_des,dq_des=dq_des,tau_ff=0.0,Kp=Kp,Kd=Kd)
                
                else:
                    raise RuntimeError("未知的机器人控制状态")
                #检查电机控制限位
                ctrl_id=motor["ctrl_id"]
                if model.actuator_ctrllimited[ctrl_id]:
                    tau_min,tau_max = model.actuator_ctrlrange[ctrl_id]

                    if(fsm.state==RobotState.STANDING and (tau < tau_min or tau > tau_max)):
                        saturated_count += 1
                    tau = np.clip(tau,tau_min,tau_max)
                    
                data.ctrl[ctrl_id] = tau

            # 推进一个仿真步
            mujoco.mj_step(model, data)

            # 每0.5秒打印一次
            if data.time - last_print_time >= 0.5:

                if fsm.state == RobotState.STANDING:
                    print(
                        f"t={data.time:.2f} "
                        f"saturated={saturated_count}/12 "
                        f"max_error={max_tracking_error:.4f}"
                    )

                last_print_time = data.time

            # 按约60Hz的频率同步Viewer
            if data.time - last_render_time >= RENDER_INTERVAL:

                viewer.sync()
                last_render_time = data.time

            # 让仿真时间尽量与真实时间同步
            elapsed = time.perf_counter() - wall_start
            remaining = data.time - elapsed

            if remaining > 0:
                time.sleep(remaining)
                
    # 打印最终状态
    print("\n=== Final Joint States ===")
    print(f"ncon = {data.ncon}")

    for i in range(data.ncon):
        contact = data.contact[i]

        geom1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1
        )

        geom2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2
        )

        print(f"{geom1_name} <-> {geom2_name}")



    for motor in motors:

        q = data.qpos[motor["qpos_id"]]
        q_target = motor["q_target"]

        print(
            f"{motor['joint_name']:20s} "
            f"q={q:+.3f} "
            f"target={q_target:+.3f} "
            f"error={q_target-q:+.4f}"
        )

if __name__ == "__main__":
    main()

