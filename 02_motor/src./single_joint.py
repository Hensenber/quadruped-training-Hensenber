import time

import mujoco
import mujoco.viewer

from model_utils import get_motor_indices


# =========================
# 1. 模型路径
# =========================

MODEL_PATH = "models/robot/single_joint.xml"


# =========================
# 2. PD 控制器
# =========================

def motor_control(q, dq, q_des, dq_des, tau_ff, Kp, Kd):

    position_error = q_des - q
    velocity_error = dq_des - dq

    tau = (
        tau_ff
        + Kp * position_error
        + Kd * velocity_error
    )

    return tau


# =========================
# 3. 主程序
# =========================

def main():

    # 加载单关节模型
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 获取关节和电机索引
    JOINT_QPOS, JOINT_QVEL, MOTOR_CTRL = get_motor_indices(
        model,
        "joint1",
        "motor1"
    )

    # 设置控制参数
    q_des = 0.3       # 目标角度，单位 rad
    dq_des = 0.0      # 目标角速度，单位 rad/s
    tau_ff = 0.0      # 前馈力矩，单位 Nm

    Kp = 20.0         # 位置比例增益
    Kd = 1.0          # 速度微分增益

    # 电机控制范围
    tau_min = -20.0
    tau_max = 20.0

    # 打印模型信息
    print("=== Single Joint Model ===")
    print(f"nq = {model.nq}")
    print(f"nv = {model.nv}")
    print(f"nu = {model.nu}")

    print("\n=== PD Parameters ===")
    print(f"q_des = {q_des}")
    print(f"Kp = {Kp}")
    print(f"Kd = {Kd}")

    # 打开可视化窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:

        last_print_time = 0.0

        # 用于让仿真速度接近真实时间
        start_wall_time = time.perf_counter()
        start_sim_time = data.time

        while viewer.is_running():

            # -------------------------
            # 读取关节状态
            # -------------------------

            q = data.qpos[JOINT_QPOS]
            dq = data.qvel[JOINT_QVEL]

            # -------------------------
            # 计算 PD 控制力矩
            # -------------------------

            tau = motor_control(
                q,
                dq,
                q_des,
                dq_des,
                tau_ff,
                Kp,
                Kd
            )

            # -------------------------
            # 力矩限幅
            # -------------------------

            tau = max(tau_min, min(tau, tau_max))

            # -------------------------
            # 发送控制指令
            # -------------------------

            data.ctrl[MOTOR_CTRL] = tau

            # -------------------------
            # 推进仿真
            # -------------------------

            mujoco.mj_step(model, data)

            # -------------------------
            # 定时打印
            # -------------------------

            if data.time - last_print_time >= 0.2:

                q_now = data.qpos[JOINT_QPOS]
                dq_now = data.qvel[JOINT_QVEL]

                print(
                    f"time={data.time:.2f}  "
                    f"q={q_now:.3f}  "
                    f"dq={dq_now:.3f}  "
                    f"q_des={q_des:.3f}  "
                    f"tau={tau:.3f}"
                )

                print(
                    f"contacts={data.ncon}  "
                    f"constraint={data.qfrc_constraint[JOINT_QVEL]:.3f}  "
                    f"actuator={data.qfrc_actuator[JOINT_QVEL]:.3f}"
                )
                last_print_time = data.time

            # 更新可视化
            viewer.sync()

            # 控制仿真运行速度
            target_wall_time = (
                start_wall_time
                + (data.time - start_sim_time)
            )

            remaining_time = (
                target_wall_time - time.perf_counter()
            )

            if remaining_time > 0:
                time.sleep(remaining_time)


if __name__ == "__main__":
    main()