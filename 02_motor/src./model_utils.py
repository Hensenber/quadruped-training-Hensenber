import mujoco

#把这个模型的关节和 actuator 情况打印出来
def print_model_info(model):
    """
    打印 MuJoCo 模型的基本信息、关节索引和执行器索引。
    用于检查模型结构。
    """
    print("=== Model Info ===")
    print("nq   =", model.nq)
    print("nv   =", model.nv)
    print("nu   =", model.nu)
    print("njnt =", model.njnt)

    print("\n=== Joints ===")

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id
        )

        qpos_id = model.jnt_qposadr[joint_id]
        qvel_id = model.jnt_dofadr[joint_id]

        print(
            f"{joint_name}: "
            f"joint_id={joint_id}, "
            f"qpos={qpos_id}, "
            f"qvel={qvel_id}"
        )

    print("\n=== Actuators ===")

    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id
        )

        print(
            f"{actuator_name}: "
            f"actuator_id={actuator_id}, "
            f"ctrl={actuator_id}"
        )


#qpos_id, qvel_id = get_joint_indices()
#告诉我 FL hip 的位置和速度去哪里读
def get_joint_indices(model, joint_name):
    """
    根据 joint 名字，返回：
        qpos 中的位置索引
        qvel 中的速度索引

    Example:
        qpos_id, qvel_id =
            get_joint_indices(model, "FL_hip_joint")
    """
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    if joint_id == -1:
        raise ValueError(f"Joint '{joint_name}' not found")

    qpos_id = model.jnt_qposadr[joint_id]
    qvel_id = model.jnt_dofadr[joint_id]

    return qpos_id, qvel_id





def get_actuator_id(model, actuator_name):
    """
    根据 actuator 名字返回 actuator ID。

    对 data.ctrl 来说：
        actuator_id 就是对应的 ctrl 索引。
    """
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name
    )

    if actuator_id == -1:
        raise ValueError(f"Actuator '{actuator_name}' not found")

    return actuator_id


#qpos_id, qvel_id, ctrl_id = get_motor_indices()
#我要控制 FL hip,直接把我要用的三个索引全给我
def get_motor_indices(model, joint_name, actuator_name):
    """
    一次获得一个电机关节控制需要的三个索引：

        qpos_id  -> data.qpos[qpos_id]
        qvel_id  -> data.qvel[qvel_id]
        ctrl_id  -> data.ctrl[ctrl_id]
    """
    qpos_id, qvel_id = get_joint_indices(
        model,
        joint_name
    )

    ctrl_id = get_actuator_id(
        model,
        actuator_name
    )

    return qpos_id, qvel_id, ctrl_id