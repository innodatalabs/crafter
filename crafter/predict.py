import time
import cv2
import numpy as np

from . import craft_utils
from . import image_utils


def get_prediction(
    image,
    craft_net,
    refine_net=None,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    long_size: int = 1280,
    poly: bool = False,
    heatmaps: bool = False,
):
    """
    Arguments:
        image: path to the image to be processed or numpy array or PIL image
        output_dir: path to the results to be exported
        craft_net: craft net model
        refine_net: refine net model
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        canvas_size: image size for inference
        long_size: desired longest image size for inference
        poly: enable polygon type
    Output:
        {"masks": lists of predicted masks 2d as bool array,
         "boxes": list of coords of points of predicted boxes,
         "boxes_as_ratios": list of coords of points of predicted boxes as ratios of image size,
         "polys_as_ratios": list of coords of points of predicted polys as ratios of image size,
         "heatmaps": visualizations of the detected characters/links,
         "times": elapsed times of the sub modules, in seconds}
    """
    t0 = time.time()

    # read/convert image
    image = image_utils.read_image(image)

    # resize
    img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(
        image, long_size, interpolation=cv2.INTER_LINEAR
    )
    ratio_h = ratio_w = 1 / target_ratio
    resize_time = time.time() - t0
    t0 = time.time()

    # preprocessing
    x = image_utils.normalizeMeanVariance(img_resized)
    x = np.transpose(x, (2, 0, 1))  # [h, w, c] to [c, h, w]
    x = np.expand_dims(x, 0)  # [c, h, w] to [b, c, h, w]
    preprocessing_time = time.time() - t0
    t0 = time.time()

    # forward pass
    y, feature = craft_net(x)
    craftnet_time = time.time() - t0
    t0 = time.time()

    # make score and link map
    score_text = y[0, 0, :, :]
    score_link = y[0, 1, :, :]

    # refine link
    if refine_net is not None:
        y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, 0, :, :]
    refinenet_time = time.time() - t0
    t0 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # get image size
    img_height = image.shape[0]
    img_width = image.shape[1]

    # calculate box coords as ratios to image size
    boxes_as_ratio = []
    for box in boxes:
        boxes_as_ratio.append(box / [img_width, img_height])
    boxes_as_ratio = np.array(boxes_as_ratio)

    # calculate poly coords as ratios to image size
    polys_as_ratio = []
    for poly in polys:
        polys_as_ratio.append(poly / [img_width, img_height])
    polys_as_ratio = np.array(polys_as_ratio, dtype=object)

    if heatmaps:
        text_score_heatmap = image_utils.cvt2HeatmapImg(score_text)
        link_score_heatmap = image_utils.cvt2HeatmapImg(score_link)
        heatmaps = {
            "text_score_heatmap": text_score_heatmap,
            "link_score_heatmap": link_score_heatmap,
        }

    postprocess_time = time.time() - t0

    times = {
        "resize_time": resize_time,
        "preprocessing_time": preprocessing_time,
        "craftnet_time": craftnet_time,
        "refinenet_time": refinenet_time,
        "postprocess_time": postprocess_time,
    }

    out = {
        "boxes": np.array(boxes, dtype=np.float32),
        "boxes_as_ratios": np.array(boxes_as_ratio, dtype=np.float32),
        "polys": polys,
        "polys_as_ratios": polys_as_ratio,
        "times": times,
    }
    if heatmaps:
        out["heatmaps"] = heatmaps

    return out
