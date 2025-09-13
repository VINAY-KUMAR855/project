from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import preprocess_input
import sqlite3
import os

app = FastAPI()

# Allow frontend (HTML/JS) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for hackathon/demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 224 

'''
CLASS_FILE = "/home/sureshsecond/Documents/sih/backend/class_names.json"
try:
    with open(CLASS_FILE, "r") as f:
        CLASS_NAMES = json.load(f)  # dictionary: {"0": "Ayrshire", "1": "Holstein"}
    print("Loaded class names:", CLASS_NAMES)
    NUM_CLASSES = len(CLASS_NAMES)
except Exception as e:
    print(f"❌Error loading class names: {e}")
    CLASS_NAMES = {}
    NUM_CLASSES = 14  # fallback
'''

def breed_name(l):
	conn=sqlite3.connect("/home/sureshsecond/Documents/sih/database/labels.db")
	cur=conn.cursor()
	cur.execute("SELECT name FROM label WHERE label = ?",(l,))
	result=cur.fetchone()
	conn.close()
	return result
def total_breeds():
	conn=sqlite3.connect("/home/sureshsecond/Documents/sih/database/labels.db")
	cur=conn.cursor()
	cur.execute("SELECT * FROM label")
	result = len(cur.fetchall())
	return result
NUM_CLASSES = total_breeds()
def create_model():

    base_model = ResNet50(
        weights='imagenet', 
        include_top=False,  
        input_shape=(224, 224, 3), 
        pooling=None        
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    # Pass through ResNet50 base
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
    x = BatchNormalization(name='batch_norm_1')(x)
    x = Dropout(0.6, name='dropout_1')(x)
    x = Dense(512,
              activation='relu',
              kernel_regularizer=l2(0.01),  # NEW: L2 regularization
              name='dense_1')(x)
    x = BatchNormalization(name='batch_norm_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    x = Dense(256,
              activation='relu',
              kernel_regularizer=l2(0.005),
              name='dense_2')(x)
    x = Dropout(0.3, name='dropout_3')(x)
    outputs = Dense(NUM_CLASSES,
                    activation='softmax',
                    kernel_regularizer=l2(0.0005),
                    name='breed_classification',
                    dtype='float32')(x)

    model = Model(inputs, outputs, name='Cattle_Buffalo_Breed_Classifier')
    return model

# Load model
try:
    MODEL = create_model()
    # Load weights instead of trying to load .h5 as complete model
    MODEL.load_weights("/home/sureshsecond/Documents/sih/backend/model/beli/best_resnet50_finetuned/model.weights.h5")
    print("✅ Model weights loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Trying alternative loading methods...")
    
    # Try loading as complete model if it's actually a complete model
    try:
        MODEL = tf.keras.models.load_model("/home/sureshsecond/Documents/sih/backend/model/beli/best_resnet50_finetuned/")
        print("✅ Complete model loaded successfully")
    except Exception as e2:
        print(f"❌ Error loading complete model: {e2}")
        MODEL = None




def read_file_as_image(data) -> np.ndarray:
    image_bytes = BytesIO(data)
    pil_image = Image.open(image_bytes)

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize for ResNet50
    pil_image = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    image_array = np.array(pil_image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    return image_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        return {"error": "Model not loaded."}
    try:
        # Read and preprocess image
        file_bytes = await file.read()
        processed_image = read_file_as_image(file_bytes)
        img_batch = np.expand_dims(processed_image, axis=0)

        # Make prediction
        predictions = MODEL.predict(img_batch, verbose=0)[0]
        print("Raw predictions:", predictions)
        print("Prediction shape:", predictions.shape)

        predicted_class_index = int(np.argmax(predictions))
        print("Predicted class index:", predicted_class_index)
        
        predicted_breed = breed_name(predicted_class_index)
        confidence = float(predictions[predicted_class_index])
        
        # Show top 3 predictions for debugging
        #top_3_indices = np.argsort(predictions)[-3:][::-1]
        #print("Top 3 predictions:")
        #for idx in top_3_indices:
        #    breed_name = CLASS_NAMES.get(str(idx), f"Class_{idx}")
        #    conf = predictions[idx]
        #    print(f"  {breed_name}: {conf:.4f}")

        return {
            "breed": predicted_breed,
            "accuracy": confidence   
        }
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

