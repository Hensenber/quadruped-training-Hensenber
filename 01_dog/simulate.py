import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene/scene.xml")
data = mujoco.MjData(model)

data.qpos[:] = np.array([
    # trunk position: x, y, z
    0.0113025555,
    0.00516957988,
    0.144941866,

    # trunk orientation: qw, qx, qy, qz
    0.999937920,
    0.000000808559040,
    0.0000122068091,
    0.0111425210,

    # FL: hip, thigh, calf
     0.501394745,
     1.60080545,
    -2.55423629,

    # FR: hip, thigh, calf
    -0.501393610,
    -1.60080306,
     2.55423900,

    # RR: hip, thigh, calf
     0.501359362,
    -1.60081606,
     2.55422931,

    # RL: hip, thigh, calf
    -0.501359963,
     1.60081862,
    -2.55422686
])

data.qvel[:] = 0.0


mujoco.mj_forward(model, data)

print("ctrl =", data.ctrl)
print("qfrc_actuator =", data.qfrc_actuator)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
