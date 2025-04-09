# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
# import numpy as np

# # Đường dẫn tới thư mục train và test
# train_dir = r'F:\Python\Project_python\nhan\Quan\data\test'
# test_dir = r'F:\Python\Project_python\nhan\Quan\data\train'
# print("Thư mục train có tồn tại không?", os.path.exists(train_dir))
# print("Thư mục test có tồn tại không?", os.path.exists(test_dir))

# # Tạo DataGenerator để tải ảnh
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='sparse'  # Nhãn là số (0-13)
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='sparse'
# )

# # Load MobileNetV2 base model
# base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model.trainable = False  # Freeze feature extractor

# # Thêm các lớp phân loại
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)
# output = Dense(14, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)

# # Biên dịch mô hình
# model.compile(optimizer=Adam(learning_rate=0.0001), 
#               loss='sparse_categorical_crossentropy', 
#               metrics=['accuracy'])
# model.summary()

# # Huấn luyện mô hình
# history = model.fit(
#     train_generator,
#     epochs=2020,
#     validation_data=test_generator
# )

# # Lưu mô hình
# model.save('bird_classifier_mobilenetv2.h5')

# # Vẽ biểu đồ kết quả
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # Hàm dự đoán ảnh mới
# def predict_bird(image_path):
#     from tensorflow.keras.preprocessing import image
#     img = image.load_img(image_path, target_size=(224, 224))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction)

#     class_id_to_name = {
#         0: 'Yellow_bellied_Flycatcher',
#         1: 'Ruby_throated_Hummingbird',
#         2: 'Rufous_Hummingbird',
#         3: 'Blue_Jay',
#         4: 'Dark_eyed_Junco',
#         5: 'Pied_Kingfisher',
#         6: 'White_breasted_Kingfisher',
#         7: 'Mallard',
#         8: 'Red_breasted_Merganser',
#         9: 'Common_Raven',
#         10: 'House_Sparrow',
#         11: 'Yellow_Warbler',
#         12: 'Cedar_Waxwing',
#         13: 'Pileated_Woodpecker'
#     }
#     bird_name = class_id_to_name[predicted_class]
#     print(f'Loài chim dự đoán: {bird_name}')
#     print(f'Xác suất cho từng loài: {prediction[0]}')

# # Dự đoán thử
# # predict_bird('test_bird.jpg')
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image

# ✅ Đường dẫn đúng: train là huấn luyện, test là kiểm tra
train_dir = r'F:\Python\Project_python\Project\assets\data\train'
test_dir = r'F:\Python\Project_python\Project\assets\data\test'

print("Thư mục train có tồn tại không?", os.path.exists(train_dir))
print("Thư mục test có tồn tại không?", os.path.exists(test_dir))

# ✅ Data Augmentation + chuẩn hóa
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# ✅ Load MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True

# Fine-tune 30 lớp cuối cùng
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(14, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ✅ Compile
model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# ✅ Callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_bird_model.keras', save_best_only=True)

# ✅ Train model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stop, checkpoint]
)

# ✅ Lưu mô hình
model.save('bird_classifier_mobilenetv2_final.keras')

# ✅ Vẽ biểu đồ kết quả
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ✅ Tạo class mapping tự động
class_id_to_name = {v: k for k, v in train_generator.class_indices.items()}

# ✅ Hàm dự đoán
def predict_bird(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    bird_name = class_id_to_name[predicted_class]
    print(f'✅ Loài chim dự đoán: {bird_name}')
    print(f'✅ Xác suất từng loài: {prediction[0]}')

# Ví dụ sử dụng:
# predict_bird('F:/Python/Project_python/test_image.png')
