"""MediaPipe 0.10.18 native file loading needs ASCII paths on Windows."""
import os
from pathlib import Path
import shutil
import tempfile

import mediapipe as mp


def create_face_mesh():
    """Return (Face Mesh graph, optional lifetime-bound temporary model folder)."""
    root = Path(mp.__file__).resolve().parent
    if os.name != "nt" or str(root).isascii():
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.6, min_tracking_confidence=0.6), None

    from mediapipe.calculators.core import constant_side_packet_calculator_pb2
    from mediapipe.calculators.tensor import inference_calculator_pb2
    from mediapipe.framework import calculator_pb2
    from mediapipe.python._framework_bindings import validated_graph_config
    from mediapipe.python.solution_base import SolutionBase

    temporary = tempfile.TemporaryDirectory(prefix="drowsiness-face-mesh-")
    try:
        if not temporary.name.isascii():
            raise RuntimeError("MediaPipe on Windows needs an ASCII TEMP directory")
        config = calculator_pb2.CalculatorGraphConfig()
        config.ParseFromString((root / "modules/face_landmark/face_landmark_front_cpu.binarypb").read_bytes())
        validated = validated_graph_config.ValidatedGraphConfig()
        validated.initialize(graph_config=config)
        config.ParseFromString(validated.binary_config)

        def model_path(relative: str) -> str:
            if not relative.startswith("mediapipe/modules/") or not relative.endswith(".tflite"):
                raise RuntimeError(f"Unexpected MediaPipe model path: {relative}")
            source = root.parent / relative
            destination = Path(temporary.name) / source.name
            shutil.copyfile(source, destination)
            return destination.as_posix()

        inference_type = inference_calculator_pb2.InferenceCalculatorOptions
        constant_type = constant_side_packet_calculator_pb2.ConstantSidePacketCalculatorOptions
        for node in config.node:
            if node.options.HasExtension(constant_type.ext):
                for packet in node.options.Extensions[constant_type.ext].packet:
                    if packet.string_value.endswith(".tflite"):
                        packet.string_value = model_path(packet.string_value)
            if node.options.HasExtension(inference_type.ext):
                options = node.options.Extensions[inference_type.ext]
                if options.model_path:
                    options.model_path = model_path(options.model_path)
            for packed in node.node_options:
                if packed.Is(inference_type.DESCRIPTOR):
                    options = inference_type()
                    packed.Unpack(options)
                    if options.model_path:
                        options.model_path = model_path(options.model_path)
                        packed.Pack(options)
        graph = SolutionBase(
            graph_config=config,
            side_inputs={"num_faces": 1, "with_attention": False, "use_prev_landmarks": True},
            calculator_params={
                "facedetectionshortrangecpu__facedetectionshortrange__facedetection__TensorsToDetectionsCalculator.min_score_thresh": 0.6,
                "facelandmarkcpu__ThresholdingCalculator.threshold": 0.6,
            },
            outputs=["multi_face_landmarks"],
        )
        return graph, temporary
    except Exception:
        temporary.cleanup()
        raise
