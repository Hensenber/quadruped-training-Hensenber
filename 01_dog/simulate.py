import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("robot/black_description.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        pass
    
