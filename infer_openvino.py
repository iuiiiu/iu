import cv2
import numpy as np
import os
from openvino.runtime import Core

# -------------------------- 配置参数（根据需求修改） -------------------------
MODEL_PATH = "openvino_ir/yolov5s"       # 模型路径（无需后缀）
IMAGE_DIR = "datasets/images"            # 输入图片目录
OUTPUT_DIR = "datasets/output"           # 输出结果目录
CLASSES_FILE = "coco.names"     # 类别名称文件（需80类）
CONF_THRESH = 0.4                         # 置信度阈值（0.4较合理，可调）
IOU_THRESH = 0.45                         # NMS阈值（合并重叠框）
INPUT_SIZE = 640                          # 模型输入尺寸（YOLOv5默认640）
# ----------------------------------------------------------------------------

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_classes():
    """加载类别名称（内置80类COCO，避免文件问题）"""
    if os.path.exists(CLASSES_FILE) and len(open(CLASSES_FILE).readlines()) == 80:
        with open(CLASSES_FILE, "r") as f:
            return [line.strip() for line in f]
    print(f"⚠️ {CLASSES_FILE} 不存在或类别数错误，使用内置80类COCO名称")
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

def preprocess_image(img):
    """图片预处理：Resize+归一化+格式转换"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_normalized = img_resized / 255.0  # 归一化到[0,1]
    return img_normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,640,640)

def postprocess_output(output, orig_h, orig_w):
    """后处理：置信度过滤+NMS+坐标转换"""
    detections = output[0]  # 模型输出形状：(25200, 85)
    
    # 过滤低置信度检测
    conf_mask = detections[:, 4] > CONF_THRESH
    detections = detections[conf_mask]
    if len(detections) == 0:
        return []
    
    # 提取边框、置信度、类别
    boxes = detections[:, :4] * np.array([orig_w/INPUT_SIZE, orig_h/INPUT_SIZE, orig_w/INPUT_SIZE, orig_h/INPUT_SIZE])
    scores = detections[:, 4]
    class_ids = np.argmax(detections[:, 5:], axis=1)
    
    # NMS去重（OpenCV实现）
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=CONF_THRESH,
        nms_threshold=IOU_THRESH
    )
    
    return [(boxes[i], scores[i], class_ids[i]) for i in indices.flatten()]

def draw_detections(img, detections, classes):
    """绘制检测框和标签（确保可见）"""
    for box, score, class_id in detections:
        x_center, y_center, w, h = box.astype(int)
        x1, y1 = max(0, x_center - w//2), max(0, y_center - h//2)  # 限制左上角坐标
        x2, y2 = min(img.shape[1], x_center + w//2), min(img.shape[0], y_center + h//2)  # 限制右下角坐标
        
        # 绘制红色边框（3像素宽）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # 绘制标签（黑色背景+白色文字）
        label = f"{classes[class_id]} {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 0, 0), -1)  # 黑色背景
        cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 白色文字
    
    return img

def process_single_image(img_path, model, classes):
    """处理单张图片：加载→预处理→推理→后处理→绘制→保存"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图片：{img_path}")
        return
    
    orig_h, orig_w = img.shape[:2]
    input_tensor = preprocess_image(img)
    output = model([input_tensor])[model.output(0)]  # 推理
    detections = postprocess_output(output, orig_h, orig_w)  # 后处理
    
    # 强制绘制（即使无目标也保存原图）
    img_with_boxes = draw_detections(img.copy(), detections, classes)
    save_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(save_path, img_with_boxes)
    print(f"✅ {os.path.basename(img_path)}：{len(detections)}个目标 → 保存至 {save_path}")

def main():
    """主函数：加载模型→批量处理图片"""
    # 1. 加载模型和类别
    try:
        core = Core()
        model = core.read_model(f"{MODEL_PATH}.xml", f"{MODEL_PATH}.bin")
        compiled_model = core.compile_model(model, "CPU")
        print(f"✅ 模型加载成功：{MODEL_PATH}.xml")
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        print("请确保openvino_ir目录下存在yolov5s.xml和yolov5s.bin")
        return
    
    classes = load_classes()
    print(f"✅ 类别加载完成：共 {len(classes)} 类")
    
    # 2. 获取图片列表
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(img_extensions)]
    if not img_files:
        print(f"❌ {IMAGE_DIR} 目录下无图片文件（支持{img_extensions}）")
        return
    
    # 3. 批量处理图片
    print(f"\n📊 开始处理 {len(img_files)} 张图片...")
    for img_file in img_files:
        process_single_image(os.path.join(IMAGE_DIR, img_file), compiled_model, classes)
    
    print(f"\n🎉 全部完成！结果保存在：{os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()