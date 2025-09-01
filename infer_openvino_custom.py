import argparse
import os
import cv2
import numpy as np
from openvino.runtime import Core

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def nms_boxes(boxes, scores, iou_threshold=0.45, score_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=scores,
        score_threshold=score_threshold,
        nms_threshold=iou_threshold
    )
    # 兼容不同 OpenCV 返回值
    if isinstance(indices, tuple) or len(indices) == 0:
        return []
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif isinstance(indices[0], (list, tuple, np.ndarray)):
        indices = [i[0] for i in indices]
    return indices
def main(opt):
    ie = Core()
    model = ie.read_model(model=opt.weights)
    compiled_model = ie.compile_model(model=model, device_name=opt.device)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    imgsz = input_layer.shape[2:]
    label_names = load_labels(opt.names)
    num_classes = len(label_names)

    img_files = [os.path.join(opt.source, f) for f in os.listdir(opt.source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    os.makedirs(opt.output, exist_ok=True)

    for img_path in img_files:
        img0 = cv2.imread(img_path)
        if img0 is None:
            print(f"Warning: could not read image {img_path}")
            continue
        img, r, (dw, dh) = letterbox(img0, new_shape=imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32) / 255.0

        preds = compiled_model([img])[output_layer]
        preds = np.squeeze(preds)
        if preds.ndim == 1:
            preds = np.expand_dims(preds, 0)

        h, w = img0.shape[:2]
        boxes, scores, class_ids = [], [], []

        for det in preds:
            if det.shape[0] <= 4:
                continue
            x1, y1, x2, y2 = det[:4]
            confs = det[4:]
            if confs.shape[0] != num_classes:
                print(f"类别数不一致：模型输出{confs.shape[0]}，labels.txt {num_classes}")
                continue
            cls = np.argmax(confs)
            conf = confs[cls]
            if conf < opt.conf_thres:
                continue
            # 坐标反算
            x1 = int(round((x1 - dw) / r))
            y1 = int(round((y1 - dh) / r))
            x2 = int(round((x2 - dw) / r))
            y2 = int(round((y2 - dh) / r))
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2-x1, y2-y1])
            scores.append(float(conf))
            class_ids.append(cls)

        indices = nms_boxes(boxes, scores, iou_threshold=0.45, score_threshold=opt.conf_thres)
        for i in indices:
            x, y, w_box, h_box = boxes[i]
            cls = class_ids[i]
            conf = scores[i]
            label = f"{label_names[cls]}:{conf:.2f}" if cls < len(label_names) else f"unknown:{conf:.2f}"
            cv2.rectangle(img0, (x, y), (x + w_box, y + h_box), (0,255,0), 2)
            cv2.putText(img0, label, (x, max(y-5,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        out_path = os.path.join(opt.output, os.path.basename(img_path))
        cv2.imwrite(out_path, img0)
        print(f"Processed: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs\train\exp41\weights\best.pt', help='OpenVINO IR model xml path')
    parser.add_argument('--source', type=str, default='aaa/images/train', help='Folder with input images')
    parser.add_argument('--output', type=str, default='datasets/output', help='Folder for output images')
    parser.add_argument('--names', type=str, default='my_labels.txt', help='Class labels file (每行一个类别名)')
    parser.add_argument('--device', type=str, default='CPU', help='Device to run inference')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    opt = parser.parse_args()
    main(opt)