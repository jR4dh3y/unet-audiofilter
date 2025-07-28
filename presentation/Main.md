---
theme: rose-pine-moon
_class: lead
paginate: true
marp: true
style: |
  section {
    font-size: 24px;
  }
  section h1 {
    font-size: 48px;
  }
  section h2 {
    font-size: 36px;
  }
  section h3 {
    font-size: 28px;
  }
  ul, ol {
    font-size: 20px;
  }
  code {
    font-size: 18px;
  }
---

# Clear Voice

## unet speech enhancement system

<small>*AI-powered noise reduction for crystal clear audio*</small>
![bg right](logo.png)

---

## **Team**

<div style="font-size: 22px;">

- **Radhey Kalra** 
- **Team Member** 
![bg right:50%](https://picsum.photos/720?image=3)

</div>

---

## **Problem Statement**

<div style="font-size: 20px;">

### The Challenge
- Background noise degrades speech quality in recordings
- Manual noise removal is time-consuming and requires expertise
- Real-time enhancement needed for communication applications

### Current Pain Points
- Poor audio quality in noisy environments
- Complex professional software requirements
- Limited real-time processing capabilities

![bg right:50%](https://fcit.usf.edu/matrix/wp-content/uploads/2016/09/50012-700.png)

</div>

---

## **Solution Offered**

<div style="font-size: 16px;">

### AI-Powered Speech Enhancement
- **U-Net Architecture**: Deep learning model for noise reduction
- **Real-time Processing**: GPU-accelerated audio enhancement  
- **Multiple Interfaces**: Web app, CLI, and Jupyter notebook workflows
- **High Quality Output**: Professional-grade noise removal

### Key Features

- STFT-based spectral processing • Chunked audio handling for memory efficiency
- Interactive visualization tools • Multiple audio format support


</div>

---

## **Data Flow Diagram**

![Data Flow center:20%  40%](dataflow_diagram.png)

<div style="font-size: 18px;">

### Processing Pipeline
1. **Input** → Noisy audio file loaded via ffmpeg
2. **Preprocess** → STFT conversion to magnitude/phase spectrograms  
3. **Enhance** → U-Net model processes magnitude spectrogram
4. **Reconstruct** → ISTFT combines enhanced magnitude with original phase
5. **Output** → Clean audio saved in desired format
6. **Evaluate** → Quality metrics and visualization tools

</div>

---

## **Technologies Used**

<div style="font-size: 18px;">

### Core Stack
- **PyTorch** | **CUDA** | **Python 3.13+** | **FFmpeg**

### Deep Learning & Audio Processing  
- **U-Net Architecture** - Encoder-decoder with skip connections | **Librosa** - Audio analysis | **NumPy** - Numerical computing

### Interface & Integration
- **Streamlit** - Web interface | **Jupyter** - Interactive development | **Matplotlib/Plotly** - Visualization

</div>

---

<div style="font-size: 19px; align:top;">

## **Outputs**

</div>

<div style="font-size: 19px;display:flex; justify-content:center; align-items:center;">

![h:300](streamlit_interface.png) ![h:300](training_curves.png)


</div>

---

## **Demo**

---

## **Future Enhancements**

<div style="font-size: 19px;">

- Real-time streaming audio enhancement
- Multi-speaker voice separation
- Mobile app development for iOS/Android
- Integration with video conferencing platforms

</div>

---

## **Q&A**
<div>

![bg right:50% h:165%](https://media1.tenor.com/m/MoCBTjJuvhIAAAAC/what-is-the-point-of-this-discord-channel-discord.gif)

</div>

---

## **Thank You**
