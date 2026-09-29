import mujoco
import numpy
import mujoco.viewer

MODEL_PATH = "models/robot/black_description.xml"

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print("nq=",model.nq)
    print("nu=",model.nu)
    print("nv=",model.nv)

    mujoco.viewer.launch(model,data)

if __name__ == "__main__":
    main()

