from typing import Optional
import os
import onnxruntime
from .predict import get_prediction
from .file_utils import export_detected_regions, export_extra_results
from .resources import res


class Craftnet:
    def __init__(self, onnx_path=None):
        if onnx_path is None:
            onnx_path = res("craftnet.onnx")
        self._onnx_session = onnxruntime.InferenceSession(onnx_path)

    def __call__(self, image):
        return self._onnx_session.run(None, {"image": image})
    
class Refinenet:
    def __init__(self, onnx_path=None):
        if onnx_path is None:
            onnx_path = res("refinenet.onnx")
        self._onnx_session = onnxruntime.InferenceSession(onnx_path)

    def __call__(self, y, feature):
        return self._onnx_session.run(None, {"y": y, "feature": feature})[0]

class Crafter:
    def __init__(
        self,
        output_dir=None,
        rectify=True,
        export_extra=True,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        long_size=1280,
        refiner=True,
        crop_type="poly",
        craftnet_onnx: Optional[str] = None,
        refinenet_onnx: Optional[str] = None,
    ):
        """
        Arguments:
            output_dir: path to the results to be exported
            rectify: rectify detected polygon by affine transform
            export_extra: export heatmap, detection points, box visualization
            text_threshold: text confidence threshold
            link_threshold: link confidence threshold
            low_text: text low-bound score
            long_size: desired longest image size for inference
            refiner: enable link refiner
            crop_type: crop regions by detected boxes or polys ("poly" or "box")
        """
        self.craft_net = None
        self.refine_net = None
        self.output_dir = output_dir
        self.rectify = rectify
        self.export_extra = export_extra
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.long_size = long_size
        self.refiner = refiner
        self.crop_type = crop_type

        # load craftnet
        self.craft_net = Craftnet(craftnet_onnx)
        # load refinernet if required
        self.refine_net = None
        if refiner:
            self.refine_net = Refinenet(refinenet_onnx)

    def detect_text(self, image):
        return self.__call__(image)

    def __call__(self, image):
        """
        Arguments:
            image: path to the image to be processed or numpy array or PIL image

        Output:
            {
                "masks": lists of predicted masks 2d as bool array,
                "boxes": list of coords of points of predicted boxes,
                "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
                "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
                "heatmaps": visualization of the detected characters/links,
                "text_crop_paths": list of paths of the exported text boxes/polys,
                "times": elapsed times of the sub modules, in seconds
            }
        """

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            long_size=self.long_size,
            heatmaps=self.export_extra,
        )

        if self.output_dir is not None:
            # export detected text regions

            # arange regions
            if self.crop_type == "box":
                regions = prediction_result["boxes"]
            elif self.crop_type == "poly":
                regions = prediction_result["polys"]
            else:
                raise TypeError("crop_type can be only 'polys' or 'boxes'")

            # export if output_dir is given
            prediction_result["text_crop_paths"] = []

            if type(image) == str:
                file_name, file_ext = os.path.splitext(os.path.basename(image))
            else:
                file_name = "image"
            exported_file_paths = export_detected_regions(
                image=image,
                regions=regions,
                file_name=file_name,
                output_dir=self.output_dir,
                rectify=self.rectify,
            )
            prediction_result["text_crop_paths"] = exported_file_paths

            # export heatmap, detection points, box visualization
            if self.export_extra:
                export_extra_results(
                    image=image,
                    regions=regions,
                    heatmaps=prediction_result["heatmaps"],
                    file_name=file_name,
                    output_dir=self.output_dir,
                )

        # return prediction results
        return prediction_result
