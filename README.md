# 🚀 Distributed Real-Time Multi-Face Emotion Recognition System  
### Hardware-Software Integration | Edge AI | IoT Networking  

---

## 📌 Overview

This project is an advanced **distributed Edge AI system** designed for real-time multi-face emotion recognition using a **Raspberry Pi 3B+** as the primary edge processing unit.

The system was developed and validated during a research internship at the **National Institute of Technology (NIT), Tiruchirappalli** under the official project title:

> **"AI Based Emotion Detection and Classification"**

The architecture leverages distributed networking to offload optical data capture while enabling low-latency AI inference and automated time-series data logging directly at the edge.

---

## 🏗️ System Architecture

### 🔹 Edge Computing
- Optimized inference using **TensorFlow Lite (TFLite Runtime)** for ARM-based architectures  
- Deployed custom CNN model converted to `.tflite` format  
- Achieved stable real-time inference on Raspberry Pi 3B+

### 🔹 Computer Vision
- Implemented **OpenCV** with Haar Cascade classifiers  
- Real-time multi-face detection in dynamic environments  
- Efficient frame preprocessing pipeline for reduced latency  

### 🔹 Distributed Networking
- Designed a **master-slave communication protocol**  
- Implemented socket-based IP video streaming  
- Offloaded optical data capture from the edge node  
- Reduced CPU utilization and thermal throttling  

### 🔹 Data Analytics
- Automated real-time logging to CSV/Excel  
- Time-series emotion classification tracking  
- Multi-user behavioral data structuring  

---

## 🧠 Software Stack

| Component | Technology |
|------------|------------|
| Language | Python 3.7+ |
| AI Framework | TensorFlow Lite (TFLite Runtime) |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Data Logging | CSV, OpenPyXL |
| Model Architecture | Custom Convolutional Neural Network (CNN) |

---

## ⚙️ Hardware Components

- **Processor:** Raspberry Pi 3B+ (ARM-v8 Architecture)  
- **Camera:** Raspberry Pi Camera Module / Network IP Camera  
- **Connectivity:** LAN / Wi-Fi via TCP/IP  
- **Deployment Mode:** Edge AI Inference  

---

## 🔄 Workflow Pipeline

1. IP Camera streams video to edge node  
2. OpenCV performs multi-face detection  
3. Cropped face regions passed to TFLite model  
4. CNN performs emotion classification  
5. Results logged with timestamps into structured CSV/Excel file  
6. System runs continuously with optimized thermal management  

---

## 🛠️ Installation & Deployment

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<Sadhurahavan5555>/Distributed-RealTime-MultiFace-Emotion-Recognition-RPi.git
cd Distributed-RealTime-MultiFace-Emotion-Recognition-RPi
```

---

### 2️⃣ Install Dependencies

Ensure Raspberry Pi is running Python 3.7+.

```bash
pip3 install -r requirements.txt
```

---

### 3️⃣ Configure Network Stream

Update the `cv2.VideoCapture()` line inside `main.py`:

```python
cap = cv2.VideoCapture("http://<IP_ADDRESS>:<PORT>/video")
```

Replace with your source device’s local IP address and streaming port.

---

### 4️⃣ Run the System

```bash
python3 main.py
```

---

## 📊 Key Achievements

### ✅ Edge AI Optimization
- Achieved real-time inference on resource-constrained ARM hardware  
- Optimized CNN for TensorFlow Lite deployment  
- Maintained stable FPS under thermal limits  

### ✅ Distributed Architecture
- Engineered master-slave networking model  
- Reduced computational bottlenecks  
- Improved system scalability  

### ✅ Temporal Data Intelligence
- Automated structured time-series emotion logging  
- Enabled multi-user behavioral analytics  
- Designed system suitable for simulation & research environments  

---

## 🔬 Research Impact

- Demonstrates feasibility of distributed Edge AI systems  
- Shows practical deployment of lightweight CNN models  
- Bridges hardware-software integration for IoT-based AI systems  

---

## 📌 Future Enhancements

- Integration with MQTT for IoT cloud communication  
- Model quantization improvements (INT8 optimization)  
- Web dashboard for live emotion analytics visualization  
- Deployment on Raspberry Pi 4 / Edge TPU  

---

## 📜 License

This project is developed for academic and research purposes.  
For reuse or extension, please provide appropriate attribution.

---

## 👨‍💻 Developed By

**Sadhurahavan**
Mechatronics Engineer
Research Intern – NIT Tiruchirappalli
