F1 Race Strategy Predictor 🏎️⏱️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## 🔍 Overview
A machine learning-powered tool that predicts optimal Formula 1 race strategies using historical telemetry data, weather conditions, and tyre degradation models.

## 🚀 Key Features
- Predicts 1-stop vs 2-stop strategies
- Tyre compound degradation visualization
- Weather impact simulation
- Historical data analysis (2022-2023 seasons)

## 🛠️ Tech Stack
| Component              | Technologies Used                             |
|------------------------|-----------------------------------------------|
| **Data Processing**    | FastF1, Pandas, NumPy                         |
| **Machine Learning**   | XGBoost, Scikit-learn                         |
| **Visualization**      | Matplotlib, Streamlit                         |
| **Deployment**         | Hugging Face Spaces, Google Colab             |

## ⚙️ Installation
```bash
# Clone repository
git clone https://huggingface.co/spaces/solosikoa/F1_Strat_Predictor

# Install dependencies
pip install -r requirements.txt
```

## 📈 Usage
### Web App Version
1. Run Streamlit app locally:
```bash
streamlit run app.py
```
2. Access via Hugging Face: [Live Demo](https://huggingface.co/spaces/solosikoa/F1_Strat_Predictor)

### Notebook Version
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yfHjxNLwwg1hlrkd3ssgE71MguS3S-dD?usp=sharing)
## 📊 Sample Output
![Strategy Comparison](assets/demo.gif)

## ✅ Advantages
- Real-time strategy simulation
- Uses actual F1 historical data
- Easy parameter tuning
- Open-source and customizable

## ⚠️ Limitations
- Limited to 2022-2023 season data
- Doesn't account for safety cars
- Simplified tyre degradation model

## 🔮 Future Roadmap
- [ ] Add safety car probability models
- [ ] Include real-time weather integration
- [ ] Support for 2024 season data
- [ ] Driver-specific performance models

## 🤝 Contributing
Pull requests welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License
MIT License - See [LICENSE](LICENSE) for details.
