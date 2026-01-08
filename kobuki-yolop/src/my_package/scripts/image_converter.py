#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2 as cv
import numpy as np
import torch
from lib.config import cfg
from lib.utils.utils import create_logger, select_device
from lib.models import get_net
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax

# Global variables
model = None
device = None
half = None

# Define augmentations using albumentations
augmentations = A.Compose([
    A.Resize(640, 640),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def apply_crf(image, seg_mask):
    # Create a dense CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], seg_mask.shape[0])

    # Get unary potentials (neg log probability)
    unary = unary_from_softmax(seg_mask)
    d.setUnaryEnergy(unary)

    # Create pairwise potentials
    feats = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10)

    # Run inference
    Q = d.inference(5)

    # Get the most probable class for each pixel
    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    return res

def detect(image, conf_thres_det, iou_thres_det, conf_thres_seg):
    global model, device, half, augmentations

    # Apply augmentations
    augmented = augmentations(image=image)
    img = augmented['image'].to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    det_out, da_seg_out, ll_seg_out = model(img)
    inf_out, _ = det_out

    # Apply NMS with the detection confidence threshold
    det_pred = non_max_suppression(inf_out, conf_thres=conf_thres_det, iou_thres=iou_thres_det, classes=None, agnostic=False)
    det = det_pred[0]

    _, _, height, width = img.shape

    # Resize segmentation outputs to match the original image size
    da_predict = da_seg_out[:, :, :height, :width]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, size=(image.shape[0], image.shape[1]), mode='bilinear')
    da_seg_mask = torch.softmax(da_seg_mask, dim=1).detach().cpu().numpy().squeeze()  # Softmax and convert to numpy

    ll_predict = ll_seg_out[:, :, :height, :width]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, size=(image.shape[0], image.shape[1]), mode='bilinear')
    ll_seg_mask = torch.softmax(ll_seg_mask, dim=1).detach().cpu().numpy().squeeze()  # Softmax and convert to numpy

    # Apply CRF
    da_seg_mask = apply_crf(image, da_seg_mask)
    ll_seg_mask = apply_crf(image, ll_seg_mask)

    # Create a writable copy of the image
    writable_image = image.copy()

    img_det = show_seg_result(writable_image, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det.shape).round()
        for *xyxy, conf, cls in reversed(det):
            if conf >= conf_thres_seg:  # Apply confidence threshold for segmentation
                label_det_pred = f'{cls} {conf:.2f}'
                plot_one_box(xyxy, img_det, label=label_det_pred, color=(255, 0, 0), line_thickness=2)

    return img_det

def callback(data):
    # Convert ROS Image message to OpenCV image
    try:
        image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    except Exception as e:
        rospy.logerr(f"Error converting ROS Image message to OpenCV image: {e}")
        return

    # Apply CLAHE
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv.merge((cl, a, b))
    image_clahe = cv.cvtColor(limg, cv.COLOR_LAB2BGR)

    # Publish the CLAHE image
    try:
        clahe_image = Image()
        clahe_image.header = data.header
        clahe_image.height, clahe_image.width = image_clahe.shape[:2]
        clahe_image.encoding = "bgr8"
        clahe_image.is_bigendian = 0
        clahe_image.step = image_clahe.shape[1] * 3
        clahe_image.data = image_clahe.tobytes()
        pub_clahe.publish(clahe_image)
    except Exception as e:
        rospy.logerr(f"Error converting CLAHE image back to ROS Image message: {e}")
        return

    # Process the CLAHE image
    processed_image = detect(image_clahe, conf_thres_det=0.3, iou_thres_det=0.5, conf_thres_seg=0.1)  # Adjusted thresholds

    # Convert OpenCV image back to ROS Image message
    try:
        ros_image = Image()
        ros_image.header = data.header
        ros_image.height, ros_image.width = processed_image.shape[:2]
        ros_image.encoding = "bgr8"
        ros_image.is_bigendian = 0
        ros_image.step = processed_image.shape[1] * 3
        ros_image.data = processed_image.tobytes()
    except Exception as e:
        rospy.logerr(f"Error converting OpenCV image back to ROS Image message: {e}")
        return

    # Publish the processed image
    pub_processed.publish(ros_image)

if __name__ == "__main__":
    rospy.init_node('image_listener', anonymous=True)
    pub_processed = rospy.Publisher('/processed_image', Image, queue_size=10)
    pub_clahe = rospy.Publisher('/clahe_image', Image, queue_size=10)  # New publisher for CLAHE images

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/naren/final_ws/src/my_package/scripts/weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf_thres_det', type=float, default=0.5, help='object detection confidence threshold')
    parser.add_argument('--iou_thres_det', type=float, default=0.1, help='object detection IoU threshold')
    parser.add_argument('--conf_thres_seg', type=float, default=0.5, help='segmentation confidence threshold')
    opt = parser.parse_args()

    # Load model
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = get_net(cfg)
    
    checkpoint = torch.load(opt.weights, map_location=device)
    print("Checkpoint keys:", checkpoint.keys())
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    if half:
        model.half()  # to FP16
    model.eval()

    cv_image = rospy.Subscriber("/my_camera/color/image_raw", Image, callback, queue_size=10)  # Changed topic here
    rospy.spin()

