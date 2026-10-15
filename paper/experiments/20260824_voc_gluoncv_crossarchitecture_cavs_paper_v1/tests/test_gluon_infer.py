import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "voc_gluon_infer_under_test", ROOT / "scripts/gluon_infer.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FakeNetwork:
    def __init__(self):
        self.loaded = None
        self.reset = None

    def load_parameters(self, path, *, ctx):
        self.loaded = (path, ctx)

    def collect_params(self):
        return self

    def reset_ctx(self, context):
        self.reset = context


class _FakeZoo:
    def __init__(self):
        self.calls = []
        self.network = _FakeNetwork()

    def get_model(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return self.network


class _FakeTensor:
    def __init__(self):
        self.expanded_axis = None

    def expand_dims(self, axis):
        self.expanded_axis = axis
        return self


class _FakeCenterNetTransform:
    def __init__(self, width, height):
        self.size = (width, height)
        self.label = None

    def __call__(self, raw, label):
        self.label = label
        return _FakeTensor(), label


class GluonInferTest(unittest.TestCase):
    def test_frozen_network_is_constructed_without_implicit_download(self):
        zoo = _FakeZoo()
        with tempfile.TemporaryDirectory() as directory:
            params = Path(directory) / "frozen.params"
            params.write_bytes(b"parameters")
            network = MODULE._construct_network(
                zoo, "detector", params=params, context="gpu0"
            )
        self.assertIs(network, zoo.network)
        self.assertEqual(
            zoo.calls,
            [("detector", {"pretrained": False, "pretrained_base": False})],
        )
        self.assertEqual(zoo.network.loaded, (str(params), None))
        self.assertEqual(zoo.network.reset, "gpu0")

    def test_center_net_uses_official_fixed_square_validation_transform(self):
        instances = []

        def factory(width, height):
            transform = _FakeCenterNetTransform(width, height)
            instances.append(transform)
            return transform

        tensor = MODULE._prepare_center_net_input(
            raw="image", input_size=512, transform_factory=factory
        )

        self.assertEqual(instances[0].size, (512, 512))
        self.assertEqual(instances[0].label.shape, (0, 6))
        self.assertEqual(tensor.expanded_axis, 0)


if __name__ == "__main__":
    unittest.main()
