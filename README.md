**🌿 Plant Disease Prediction Using Deep Learning**

This project implements a Plant Disease Prediction System using Convolutional Neural Networks (CNN) with Transfer Learning (MobileNetV2).
The model is trained on the PlantVillage dataset and can predict plant diseases from leaf images, displaying both true labels and predicted labels.

The entire project is designed to run on Google Colab and loads the dataset directly from Kaggle.

**📌 Features**

Uses PlantVillage Dataset from Kaggle
Transfer Learning with MobileNetV2
Automatic train / validation split
Predicts disease from random leaf images
**Displays:**
Leaf image
True label
Predicted label
Easy to extend for custom image prediction

**🗂 Dataset**

Source: Kaggle – PlantVillage Dataset
Structure used in this project:

plant_village/
└── PlantVillage/
    ├── Pepper__bell___Bacterial_spot
    ├── Pepper__bell___healthy
    ├── Potato___Early_blight
    ├── Potato___Late_blight
    ├── Potato___healthy
    ├── Tomato___Early_blight
    ├── Tomato___Late_blight
    ├── Tomato___healthy
    └── ...

Each folder represents one class (plant + disease).

**🛠 Technologies Used**

Python
TensorFlow / Keras
MobileNetV2
NumPy
Matplotlib
Google Colab
Kaggle API

**🚀 How to Run the Project (Google Colab)**

**1️⃣ Upload Kaggle API Key**

Upload your kaggle.json file to Colab.
!pip install kaggle

**2️⃣ Download and Extract Dataset**

!kaggle datasets download -d emmarex/plantdisease
!unzip plantdisease.zip -d plant_village

**3️⃣ Set Dataset Path**

train_path = "plant_village/PlantVillage"

**4️⃣ Train the Model**

Image size: 224 x 224
Batch size: 32
Optimizer: Adam
Loss: Categorical Crossentropy
Epochs: 5 (can be increased)

**5️⃣ Predict Random Leaf Image**

The prediction function:
Selects a random image
Predicts the disease

Displays image with true & predicted labels

predict_random_image()

**🧠 Model Architecture**

Base Model: MobileNetV2 (pre-trained on ImageNet)
Frozen convolutional layers
Global Average Pooling
Dropout (0.2)
Dense Softmax output layer

📊** Output Example**
True Label: Tomato___Late_blight
Predicted Label: Tomato___Late_blight

Displayed together with the leaf image.

**📈 Possible Improvements**

Fine-tune MobileNetV2 layers
Increase epochs for better accuracy
Add confusion matrix & classification report
Deploy as a web or mobile app
Add real-time camera prediction

**📄 License**

This project is for educational and research purposes.
Dataset credit goes to the PlantVillage Project and Kaggle.

**🙌 Acknowledgements**

Kaggle
PlantVillage Dataset
TensorFlow & Keras Team
Google Colab
