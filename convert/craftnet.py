from craft_text_detector import load_craftnet_model
import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np

def craftnet_to_onnx(craftnet, output_fname, *, cuda=False, width=224, height=340):
    image = torch.rand(1, 3, height, width)
    with torch.no_grad():
        y, feature = craftnet(image)

    torch.onnx.export(
        craftnet,
        (image,),
        output_fname,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["y", 'feature'],
        dynamic_axes={
            "image": {
                2: "height",
                3: "width",
            },
            "y": {
                2: "height2",
                3: "width2",
            },
            "feature": {
                2: "height2",
                3: "width2",
            }
        },
        verbose=False,
    )
    onnx_model = onnx.load(output_fname)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(output_fname)

    ort_inputs = {ort_session.get_inputs()[0].name: image.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(y.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(feature.numpy(), ort_outs[1], rtol=1e-03, atol=1e-05)


craftnet = load_craftnet_model(cuda=False)
craftnet.eval()
craftnet_to_onnx(craftnet, "crafter/resources/craftnet.onnx")
