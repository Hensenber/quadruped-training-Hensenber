import mujoco
import mujoco.viewer

from model_utils import print_model_info, get_motor_indices

MODEL_PATH = "models/robot/black_description.xml"

def motor_control(q,dq,q_des,dq_des,tau_ff,Kp,Kd):
    tau = tau_ff + Kp * (q_des - q) + Kd * (dq_des - dq)
    return tau


def main():
    #加载模型
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    #获取索引
    FL_HIP_QPOS,FL_HIP_QVEL,FL_HIP_CTRL = get_motor_indices(model,"FL_hip_joint","FL_hip_joint_motor")

    #设置5个可变参数
    q_des = 0.3
    dq_des = 0.0
    tau_ff = 0.0
    Kp = 20.0
    Kd = 1.0

    #打开Viwer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print_time = 0.0

        while viewer.is_running():

            # 读取关节状态
            q = data.qpos[FL_HIP_QPOS]
            dq = data.qvel[FL_HIP_QVEL]

            # 计算控制力矩
            tau = motor_control(
                q, dq,
                q_des, dq_des,
                tau_ff,
                Kp, Kd
            )

        # 输出控制力矩
            data.ctrl[FL_HIP_CTRL] = tau

        # 仿真推进
            mujoco.mj_step(model, data)

        # 每 0.2 秒打印一次
            if data.time - last_print_time >= 0.2:
                print(
                f"time={data.time:.2f}  "
                f"q={q:.3f}  "
                f"dq={dq:.3f}  "
                f"q_des={q_des:.3f}  "
                f"tau={tau:.3f}"
            )

                last_print_time = data.time

        # 更新画面
            viewer.sync()

if __name__ == "__main__":
    main()