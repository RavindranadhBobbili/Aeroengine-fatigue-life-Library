# Aeroengine-fatigue-life-Library
Machine learning library for predicting fatigue life (cycles to failure) of aeroengine alloys including GH4169, Inconel 718, Ti-6Al-4V, RR1000, and Waspaloy. Features Random Forest, XGBoost, and Neural Network models trained on real literature data from 200+ research articles. Includes FastAPI server for production deployment.


machine-learning
materials-science
fatigue-analysis
aeroengine
alloy-design
python-library
predictive-maintenance
random-forest
xgboost
neural-networks
fastapi
data-science
aerospace
mechanical-engineering
materials-informatics
scientific-computing
research-tool
industrial-ai



# 🚀 Aeroengine Fatigue Life

## Machine Learning for Fatigue Life Prediction of Aeroengine Alloys

[![PyPI version](https://badge.fury.io/py/aeroengine-fatigue-life.svg)](https://pypi.org/project/aeroengine-fatigue-life/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://aeroengine-fatigue-life.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://pepy.tech/badge/aeroengine-fatigue-life)](https://pepy.tech/project/aeroengine-fatigue-life)

**Predict fatigue life (cycles to failure) of aeroengine alloys using state-of-the-art machine learning.**

---

## 🎯 What is Aeroengine Fatigue Life?

Aeroengine Fatigue Life is a production-ready Python library that predicts the **fatigue life** of aeroengine alloys using machine learning. It's designed for materials scientists, engineers, and researchers who need rapid, accurate fatigue predictions without expensive experimental testing.

### Key Capabilities:
- 🔬 **Predict cycles to failure** for 6+ aeroengine alloys
- 🤖 **3 ML models**: Random Forest, XGBoost, Neural Network  
- 📊 **Real data**: Trained on 200+ research articles
- 🚀 **Production ready**: FastAPI server included
- ⚡ **Ultra-fast**: 2-5ms predictions

---

## 📊 Supported Alloys

| Alloy | Application | Data Points |
|-------|-------------|-------------|
| **GH4169** | Turbine disks | 5 |
| **Inconel 718** | Turbine blades | 4 |
| **GH4586** | Rocket engines | 3 |
| **Ti-6Al-4V** | Compressor blades | 4 |
| **RR1000** | Advanced Ni alloy | 4 |
| **Waspaloy** | Turbine disks | 4 |

---

## 🚀 Quick Install & Use

```bash
# Install
pip install aeroengine-fatigue-life

# Use in Python
from aeroengine_fatigue_life import create_predictor
predictor = create_predictor()
prediction = predictor.predict(your_data)

# Or run demo
aeroengine-fatigue-life

@software{RAVINDRANADHBOBBILI2026aeroengine,
  author = {RAVINDRANADH BOBBILI},
  title = {Aeroengine Fatigue Life: ML for Fatigue Prediction},
  year = {2026},
  url = {https://github.com/RAVINDRANADHBOBBILI/Aeroengine-fatigue-life-Library},
}

MIT License

Copyright (c) 2026 RAVINDRANADH BOBBILI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
