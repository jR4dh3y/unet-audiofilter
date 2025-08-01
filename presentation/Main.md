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

## U-Net speech enhancement system

<small>*AI-powered noise reduction for crystal clear audio*</small>
![bg right](logo.png)

---

## **Team Members**

<div style="font-size: 22px;">

- **Radhey Kalra** 
- **Aabish Malik** 
![bg right:50%](team.png)

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

![bg right:50% 140%](gif.gif)

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
<div style="font-size: 15px;">


### Processing Pipeline
1. **Input** → Noisy audio file loaded via ffmpeg
2. **Preprocess** → STFT conversion to magnitude/phase spectrograms  
3. **Enhance** → U-Net model processes magnitude spectrogram
4. **Reconstruct** → ISTFT combines enhanced magnitude with original phase
5. **Output** → Clean audio saved in desired format

![fit](dataflow_simple.png)
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

<div style="font-size: 19px;display:flex; justify-content:center; align-items:center; flex-direction:column;" >

### Audio Comparison

**Noisy:**
<audio controls>
  <source src="../results/comparison/p232_010_noisy.wav" type="audio/wav">
</audio>


**Enhanced:**
<audio controls>
  <source src="../results/comparison/p232_010_enhanced.wav" type="audio/wav">
</audio>


**Clean:**
<audio controls>
  <source src="../results/comparison/p232_010_clean.wav" type="audio/wav">
</audio>


![bg right fit](audio_comparison.png)

</div>


</div>

---

## **Demo**

---

## **Future Enhancements**

<div style="font-size: 19px;">

![bg right:50% ](future.png)

- Real-time streaming audio enhancement
- Multi-channel voice separation
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
