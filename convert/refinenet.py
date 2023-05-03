from craft_text_detector import load_refinenet_model
import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np

def refinenet_to_onnx(refinenet, output_fname, *, width=224, height=340, verbose=False):
    y = torch.rand(1, 2, height, width)
    feature = torch.rand(1, 32, height, width)
    with torch.no_grad():
        y_refined = refinenet(y, feature)

    torch.onnx.export(
        refinenet,
        (y, feature),
        output_fname,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["y", "feature"],
        output_names=["y_refined"],
        dynamic_axes={
            "y": {
                2: "height",
                3: "width",
            },
            "feature": {
                2: "height",
                3: "width",
            },
            "y_refined": {
                2: "height",
                3: "width",
            }
        },
        verbose=verbose,
    )
    onnx_model = onnx.load(output_fname)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(output_fname)

    ort_inputs = {
        ort_session.get_inputs()[0].name: y.numpy(),
        ort_session.get_inputs()[1].name: feature.numpy(),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(y_refined.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)


refinenet = load_refinenet_model(cuda=False)
refinenet.eval()
refinenet_to_onnx(refinenet, "crafter/resources/refinenet.onnx")
