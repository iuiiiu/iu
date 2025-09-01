import cv2
import numpy as np
import os
from openvino.runtime import Core

# -------------------------- 配置参数（重点修改这里） -------------------------
MODEL_PATH = "openvino_ir/yolov5s"    # OpenVINO模型路径（无需后缀）
IMAGE_DIR = "aaa\images\train"         # 待检测图片目录
OUTPUT_DIR = "datasets/output"        # 结果保存目录
CLASS_FILE = "my_labels.txt" # 自定义类别文件（每行一个类别名）
CONF_THRESH = 0.4                     # 置信度阈值（0.4较合理）
INPUT_SIZE = 640                      # 模型输入尺寸（与训练时一致）
# ----------------------------------------------------------------------------

def load_classes():
    """加载自定义类别（从CLASS_FILE读取）"""
    if not os.path.exists(CLASS_FILE):
        print(f"❌ 类别文件不存在：{CLASS_FILE}")
        exit()
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"✅ 已加载 {len(classes)} 个自定义类别：{classes}")
    return classes

def preprocess(img):
    """预处理：Resize+归一化+格式转换"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB
    img_resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img_normalized = img_resized / 255.0  # 归一化到[0,1]
    return img_normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1,3,640,640)

def postprocess(output, orig_h, orig_w):
    """后处理：提取检测框+坐标转换+NMS去重"""
    detections = output[0]  # 模型输出形状：(25200, 85)
    
    # 1. 过滤低置信度检测
    conf_mask = detections[:, 4] > CONF_THRESH
    detections = detections[conf_mask]
    if len(detections) == 0:
        return []
    
    # 2. 提取坐标和类别
    boxes = detections[:, :4] * np.array([orig_w/INPUT_SIZE, orig_h/INPUT_SIZE, orig_w/INPUT_SIZE, orig_h/INPUT_SIZE])
    scores = detections[:, 4]
    class_ids = np.argmax(detections[:, 5:], axis=1)  # 类别概率最大的索引
    
    # 3. NMS去重（合并重叠框）
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=CONF_THRESH,
        nms_threshold=0.45
    )
    
    return [(boxes[i], scores[i], class_ids[i]) for i in indices.flatten()]

def draw_boxes(img, detections, classes):
    """绘制检测框和标签（确保可见）"""
    for box, score, class_id in detections:
        x_center, y_center, w, h = box.astype(int)
        x1, y1 = max(0, x_center - w//2), max(0, y_center - h//2)  # 左上角（限制在图片内）
        x2, y2 = min(img.shape[1], x_center + w//2), min(img.shape[0], y_center + h//2)  # 右下角
        
        # 绘制红色边框（线宽3）
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # 绘制标签（黑色背景+白色文字）
        label = f"{classes[class_id]} {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1-label_h-10), (x1+label_w, y1), (0, 0, 0), -1)  # 黑色背景
        cv2.putText(img, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 白色文字
    
    return img

def process_single_image(img_path, model, classes):
    """处理单张图片：加载→预处理→推理→后处理→绘制→保存"""
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 跳过无效图片：{img_path}")
        return
    
    # 推理
    orig_h, orig_w = img.shape[:2]
    input_tensor = preprocess(img)
    output = model([input_tensor])[model.output(0)]  # 模型输出
    
    # 后处理+绘制
    detections = postprocess(output, orig_h, orig_w)
    img_with_boxes = draw_boxes(img.copy(), detections, classes)
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(save_path, img_with_boxes)
    print(f"✅ {os.path.basename(img_path)}：{len(detections)}个目标 → 保存至 {save_path}")

def main():
    """主函数：加载模型→批量处理图片"""
    # 1. 加载模型和类别
    try:
        core = Core()
        model = core.read_model(f"{MODEL_PATH}.xml", f"{MODEL_PATH}.bin")
        compiled_model = core.compile_model(model, "CPU")  # 编译到CPU（可选GPU）
        print(f"✅ 成功加载OpenVINO模型：{MODEL_PATH}.xml")
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        print("请检查openvino_ir目录下是否有yolov5s.xml和yolov5s.bin")
        return
    
    classes = load_classes()
    
    # 2. 批量处理图片
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(img_extensions)]
    if not img_files:
        print(f"❌ 未找到图片：{IMAGE_DIR} 目录下需有{img_extensions}格式图片")
        return
    
    print(f"\n📊 开始处理 {len(img_files)} 张图片...")
    for img_file in img_files:
        process_single_image(os.path.join(IMAGE_DIR, img_file), compiled_model, classes)
    
    print(f"\n🎉 全部完成！结果保存在：{os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()