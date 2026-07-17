import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. إعدادات واجهة الموقع
st.set_page_config(page_title="AI Fashion Classifier", page_icon="👕")
st.title("👕 AI Fashion Classifier")
st.write("ارفع صورة أي قطعة ملابس (تيشيرت، بنطلون، جزمة، شنطة...) والسيستم هيتعرف عليها فوراً.")

# 2. القاموس البشري
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. تحميل الموديل
@st.cache_resource 
def load_model():
    # تأكد إن اسم الموديل هنا مطابق للي حفظته في الـ Jupyter
    return tf.keras.models.load_model('fashion_model.h5')

model = load_model()

# 4. زرار رفع الصورة
uploaded_file = st.file_uploader("اختار صورة الملابس...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة للعميل
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_container_width=True)
    
    st.write("جاري تحليل التفاصيل... ⏳")
    
    # 5. تجهيز الصورة (العمليات اللي الموديل بيحبها)
    # تحويل الصورة لأبيض وأسود
    img_array = np.array(image.convert('L')) 
    
    # تصغيرها لمقاس 28 في 28 بيكسل
    img_resized = cv2.resize(img_array, (28, 28)) 
    
    # عكس الألوان (لأن داتا التدريب كانت الهدوم بيضاء والخلفية سوداء)
    # الخطوة دي بتعلي الدقة جداً في الصور العادية
    img_inverted = cv2.bitwise_not(img_resized)
    
    # تظبيط الأبعاد والألوان
    img_reshaped = img_inverted.reshape(-1, 28, 28, 1) / 255.0 
    
    # 6. التوقع وإعلان النتيجة
    predictions = model.predict(img_reshaped)
    predicted_label = np.argmax(predictions)
    
    st.write("---")
    st.success(f"✅ النتيجة: القطعة دي عبارة عن ( {class_names[predicted_label]} )")