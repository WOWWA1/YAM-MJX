from pathlib import Path

import mujoco
import mujoco.viewer


def main() -> None:
    xml = Path(
        "/home/drosakis/yam-mujocoplayground/mujoco_playground/"
        "mujoco_playground/external_deps/mujoco_menagerie/i2rt_yam/scene.xml"
    )

    print("XML path:", xml)
    print("Exists:", xml.exists())

    if not xml.exists():
        raise FileNotFoundError(f"Could not find YAM scene.xml at: {xml}")

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    print("Launching YAM viewer...")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

