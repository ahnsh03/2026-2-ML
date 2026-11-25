"""Validate emitted packet signs and enum/payload separation without ROS or a simulator."""

import importlib.util, json, math, struct, sys, types, unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch


def bridge_module():
    modules = {}
    for package, names in [
        ("diagnostic_msgs", ["DiagnosticArray", "DiagnosticStatus", "KeyValue"]),
        ("geometry_msgs", ["Vector3"]),
        (
            "morai_msgs",
            [
                "CollisionData",
                "CtrlCmd",
                "EgoVehicleStatus",
                "ObjectStatus",
                "ObjectStatusList",
            ],
        ),
    ]:
        modules[package] = types.ModuleType(package)
        msg = types.ModuleType(package + ".msg")
        for name in names:
            setattr(msg, name, type(name, (), {}))
        modules[package + ".msg"] = msg
    modules["rospy"] = types.ModuleType("rospy")
    path = (
        Path(__file__).resolve().parents[1]
        / "ros/src/data_collection/scripts/morai_udp_bridge.py"
    )
    spec = importlib.util.spec_from_file_location("bridge_steering_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class SteeringTests(unittest.TestCase):
    def test_both_message_layouts_and_rear_extension_invert_once(self):
        module = bridge_module()
        for inverted in (False, True):
            for payload_bytes in (23, 27):
                for beta in (False, True):
                    bridge = module.MoraiUdpBridge.__new__(module.MoraiUdpBridge)
                    bridge.max_steer_deg = 40
                    bridge.invert_steer = inverted
                    bridge.cmd_bytes = payload_bytes
                    bridge.processing_ms = {"cmd": deque()}
                    bridge.stats = {"cmd": 0}
                    bridge.last_activity_monotonic = {}
                    bridge.cmd_ip = "127.0.0.1"
                    bridge.cmd_port = 0
                    packets = []
                    bridge.cmd_sock = types.SimpleNamespace(
                        sendto=lambda p, addr: packets.append(p)
                    )
                    fields = dict(
                        longlCmdType=2,
                        velocity=0.0,
                        acceleration=0.0,
                        accel=0.0,
                        brake=1.0,
                    )
                    fields.update(
                        dict(steering=math.radians(10))
                        if beta
                        else dict(
                            front_steer=math.radians(10), rear_steer=math.radians(-5)
                        )
                    )
                    bridge.on_ctrl_cmd(types.SimpleNamespace(**fields))
                    packet = packets[0]
                    self.assertEqual(
                        struct.unpack(
                            "<i", packet[len(module.CMD_HDR) : len(module.CMD_HDR) + 4]
                        )[0],
                        payload_bytes,
                    )
                    values = struct.unpack(
                        module.CMD_FMT_27 if payload_bytes == 27 else module.CMD_FMT_23,
                        packet[-payload_bytes - 2 : -2],
                    )
                    self.assertEqual(values[:3], (2, 4, 2))
                    self.assertAlmostEqual(
                        values[-2] if payload_bytes == 27 else values[-1],
                        -0.25 if inverted else 0.25,
                    )
                    if payload_bytes == 27:
                        self.assertAlmostEqual(
                            values[-1], 0 if beta else (0.125 if inverted else -0.125)
                        )


class SimulatorProfileTests(unittest.TestCase):
    def test_release_defaults_match_controller_import_and_packet_size(self):
        import xml.etree.ElementTree as ET
        root = Path(__file__).resolve().parents[1]
        profiles = json.loads((root / "config/simulator_release_profiles.json").read_text())["profiles"]
        launch = ET.parse(root / "ros/src/data_collection/launch/morai_udp_bridge.launch").getroot()
        defaults = {item.attrib["name"]: item.attrib.get("default") for item in launch.findall("arg")}
        for release, payload, total in [("26r1", 27, 59), ("molitcomp03", 23, 55)]:
            expression = defaults["cmd_payload_bytes"][7:-1]
            resolved = eval(expression, {"__builtins__": {}}, {"arg": lambda name: release})
            self.assertEqual(resolved, payload)
            self.assertEqual(profiles[release]["command_payload_bytes"], payload)
            self.assertEqual(profiles[release]["network_receive_buffer_size"], total)
            self.assertEqual(len(b"#MoraiCtrlCmd$") + 4 + 12 + payload + 2, total)
        network = json.loads((root / "config/morai/network_set_local.json").read_text())
        controllers = [item for ego in network["egoNetworkData"] for item in ego["listEgoNetworkInfo"] if item["Division_TYPE"] == 3]
        self.assertEqual(len(controllers), 1)
        controller = controllers[0]
        self.assertEqual(controller["PUBSUB_TYPE"], 771)
        self.assertEqual(controller["EgoCtrlConfig"]["_PUBSUB_TYPE"], 771)
        self.assertEqual(controller["EgoCtrlConfig"]["egoCtrlMode"], 3)
        self.assertEqual(controller["EgoCtrlConfig"]["udpConfig"]["receiveBufferSize"], 59)


if __name__ == "__main__":
    unittest.main()
