# Embedded ML Memory Optimization Techniques

A side-by-side evaluation of system‐level and model‐centric memory‐optimization techniques for real‑time human activity recognition (HAR) on resource‑constrained edge platforms, using the WISDM dataset. This project re‑implements and compares five embedded‑inspired optimizations and four ML‑centric methods, culminating in a hybrid pipeline that balances memory footprint, inference latency, and predictive accuracy.

---

## 📋 Table of Contents

- [🚀 Project Overview](#-project-overview)  
- [🔧 Features](#-features)  
- [⚙️ Getting Started](#️-getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [🗂 Repository Structure](#-repository-structure)  
- [🛠 Implemented Techniques](#-implemented-techniques)  
  - [System‑Level Optimizations](#system‑level-optimizations)  
  - [Model‑Centric Methods](#model‑centric-methods)  
  - [Hybrid Pipeline](#hybrid-pipeline)  
- [🧪 Experimental Setup](#-experimental-setup)  
- [📈 Results & Evaluation](#-results--evaluation)  
- [🤝 Contributing](#-contributing)  
- [📄 License](#-license)  
- [📚 References](#-references)  

---

## 🚀 Project Overview

Edge and IoT devices often have stringent constraints on memory (≤4 MB), compute (≤200 MHz), and power (≤50 mW), making deployment of modern ML models challenging. This repo implements:

1. **System‑level code optimizations** (static memory pruning, code‑hierarchy flattening, lazy loading, etc.)  
2. **Model‑centric methods** (int8 quantization, structured pruning, dynamic loading, PCA, contrastive encoding)  
3. **A hybrid pipeline** combining both approaches for the best trade‑off between resource efficiency and accuracy.

---

## 🔧 Features

- 🚀 Ultra‑low memory footprint (down to 0.08 MB)  
- ⚡ Real‑time inference latency (as low as 0.03 s)  
- 📊 High HAR accuracy (up to 93.7 %)  
- 🔄 Modular pipelines for easy swapping of optimization techniques  
- 📑 Automated scripts for reproducible experiments  

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.12  
- PyTorch 2.x  
- scikit‑learn  
- numpy, pandas, matplotlib  


```bash
# Clone the repo
git clone https://github.com/yourusername/embedded-ml-memory-optimization.git
cd embedded-ml-memory-optimization

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
