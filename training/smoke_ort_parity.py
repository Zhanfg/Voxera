from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import numpy as np
import torch

from training.condition_fusion import ConditionFusion, ConditionFusionConfig
from training.contentnet import CausalContentNet, ContentNetConfig
from training.decoder import DecoderNet, DecoderNetConfig
from training.lite_vocoder import LiteVocoder, LiteVocoderConfig
from training.timbrenet import TimbreNet, TimbreNetConfig


ATOL = 3e-5
RTOL = 2e-4


def _session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )


def _assert_close(name: str, expected: torch.Tensor, actual: np.ndarray) -> float:
    expected_np = expected.detach().cpu().numpy()
    np.testing.assert_allclose(
        actual,
        expected_np,
        rtol=RTOL,
        atol=ATOL,
        err_msg=name,
    )
    return float(np.max(np.abs(actual - expected_np))) if actual.size else 0.0


def main() -> int:
    export_bundle = importlib.import_module("training.export_bundle")

    torch.manual_seed(109)
    models = export_bundle.CoreModels(
        contentnet=CausalContentNet(ContentNetConfig()).eval(),
        timbrenet=TimbreNet(TimbreNetConfig()).eval(),
        condition_fusion=ConditionFusion(ConditionFusionConfig()).eval(),
        decoder=DecoderNet(DecoderNetConfig()).eval(),
        lite_vocoder=LiteVocoder(LiteVocoderConfig()).eval(),
    )

    errors: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="voxera-ort-parity-") as temporary:
        root = Path(temporary)
        export_bundle._export_contentnet(models.contentnet, root / "contentnet.onnx", 4, 17)
        export_bundle._export_timbrenet(models.timbrenet, root / "timbrenet.onnx", 8, 17)
        export_bundle._export_fusion(models.condition_fusion, root / "condition_fusion.onnx", 2, 17)
        export_bundle._export_decoder(models.decoder, root / "decodernet.onnx", 2, 17)
        export_bundle._export_vocoder(models.lite_vocoder, root / "lite_vocoder.onnx", 4, 17)

        with torch.no_grad():
            features = torch.randn(1, 7, 80)
            cache = models.contentnet.initial_cache(1)
            pt_content, pt_content_cache = models.contentnet.forward_chunk(
                features,
                cache,
            )
            ort_content, ort_content_cache = _session(root / "contentnet.onnx").run(
                None,
                {
                    "features": features.numpy(),
                    "cache": cache.numpy(),
                },
            )
            errors["content"] = _assert_close(
                "contentnet.content",
                pt_content,
                ort_content,
            )
            errors["content_cache"] = _assert_close(
                "contentnet.next_cache",
                pt_content_cache,
                ort_content_cache,
            )

            reference = torch.randn(1, 13, 80)
            pt_speaker = models.timbrenet(reference)
            (ort_speaker,) = _session(root / "timbrenet.onnx").run(
                None,
                {"features": reference.numpy()},
            )
            errors["speaker"] = _assert_close(
                "timbrenet.speaker",
                pt_speaker,
                ort_speaker,
            )

            content = torch.randn(1, 5, 256)
            pitch = torch.randn(1, 5, 3)
            speaker = torch.randn(1, 256)
            pt_condition, _ = models.condition_fusion(content, pitch, speaker)
            (ort_condition,) = _session(root / "condition_fusion.onnx").run(
                None,
                {
                    "content": content.numpy(),
                    "pitch": pitch.numpy(),
                    "speaker": speaker.numpy(),
                },
            )
            errors["condition"] = _assert_close(
                "condition_fusion.condition",
                pt_condition,
                ort_condition,
            )

            conditions = torch.randn(1, 3, 256)
            decoder_cache = models.decoder.initial_cache(1)
            pt_mel, pt_decoder_cache = models.decoder.forward_chunk(
                conditions,
                decoder_cache,
            )
            ort_mel, ort_decoder_cache = _session(root / "decodernet.onnx").run(
                None,
                {
                    "conditions": conditions.numpy(),
                    "cache": decoder_cache.numpy(),
                },
            )
            errors["mel"] = _assert_close("decodernet.mel", pt_mel, ort_mel)
            errors["decoder_cache"] = _assert_close(
                "decodernet.next_cache",
                pt_decoder_cache,
                ort_decoder_cache,
            )

            mel = torch.randn(1, 7, 80)
            vocoder_cache = models.lite_vocoder.initial_cache(1)
            pt_magnitude, pt_phase, pt_vocoder_cache = (
                models.lite_vocoder.forward_chunk(mel, vocoder_cache)
            )
            ort_magnitude, ort_phase, ort_vocoder_cache = _session(
                root / "lite_vocoder.onnx"
            ).run(
                None,
                {
                    "mel": mel.numpy(),
                    "cache": vocoder_cache.numpy(),
                },
            )
            errors["magnitude"] = _assert_close(
                "lite_vocoder.log_magnitude",
                pt_magnitude,
                ort_magnitude,
            )
            errors["phase"] = _assert_close(
                "lite_vocoder.phase",
                pt_phase,
                ort_phase,
            )
            errors["vocoder_cache"] = _assert_close(
                "lite_vocoder.next_cache",
                pt_vocoder_cache,
                ort_vocoder_cache,
            )

    for name, error in errors.items():
        print(f"{name}_max_abs_error={error:.9g}")
    print(f"checked_tensors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
