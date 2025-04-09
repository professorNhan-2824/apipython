import numpy as np
import tensorflow as tf
from PIL import Image

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path=r"F:\Python\Project_python\nhan\bird_classifier_mobilenetv2_final.tflite")
interpreter.allocate_tensors()

# Lấy thông tin input và output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lấy kích thước input từ model
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Ánh xạ nhãn với tên loài chim
class_names = {
    0: 'Đớp ruồi bụng vàng',  # Yellow_bellied_Flycatcher
    1: 'Chim ruồi họng đỏ',   # Ruby_throated_Hummingbird
    2: 'Chim ruồi hung',      # Rufous_Hummingbird
    3: 'Giẻ cùi lam',         # Blue_Jay
    4: 'Chích tối mắt',       # Dark_eyed_Junco
    5: 'Bói cá mào',          # Pied_Kingfisher
    6: 'Bói cá bụng trắng',   # White_breasted_Kingfisher
    7: 'Vịt cổ xanh',         # Mallard
    8: 'Vịt Merganser ngực đỏ', # Red_breasted_Merganser
    9: 'Quạ đen thường',      # Common_Raven
    10: 'Chim sẻ nhà',        # House_Sparrow
    11: 'Chích vàng',         # Yellow_Warbler
    12: 'Chim bách thanh tuyết tùng', # Cedar_Waxwing
    13: 'Gõ kiến đầu đỏ'      # Pileated_Woodpecker
}

# Hàm xử lý và dự đoán ảnh
def predict_image(image_path):
    # Đọc và tiền xử lý ảnh
    img = Image.open(image_path).convert('RGB')
    img = img.resize((width, height))
    img_array = np.array(img, dtype=np.float32)
    
    # Chuẩn hóa ảnh (điều chỉnh theo cách bạn đã train model)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Chạy inference
    interpreter.invoke()
    
    # Lấy kết quả
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Lấy class dự đoán
    predicted_class_idx = np.argmax(output_data[0])
    confidence = output_data[0][predicted_class_idx]
    
    # Lấy tên loài chim từ class_id
    predicted_bird_name = class_names[predicted_class_idx]
    english_name = predicted_bird_name.split('# ')[1] if '# ' in predicted_bird_name else ""
    
    return predicted_class_idx, predicted_bird_name, confidence

image_path = r"F:\Python\Project_python\nhan\7.png"
class_id, bird_name, confidence = predict_image(image_path)
print(f"Loài chim dự đoán: {bird_name} (ID: {class_id})")
print(f"Độ tin cậy: {confidence:.2f}")