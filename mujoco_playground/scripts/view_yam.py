import mujoco
import mujoco.viewer

from yam_assets import require_yam_file


def main() -> None:
    xml = require_yam_file("scene.xml")

    print("XML path:", xml)

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    print("Launching YAM viewer...")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
