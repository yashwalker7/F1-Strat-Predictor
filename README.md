# F1 Race Strategy Predictor 🏎️⏱️

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

## 📊 Visual Demonstration
<div align="center" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0;">
  <a href="assets/a.png" target="_blank">
    <img src="assets/a.png" alt="Strategy Comparison" loading="lazy" style="width:100%; border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
  </a>
  <a href="assets/b.png" target="_blank">
    <img src="assets/b.png" alt="Weather Impact" loading="lazy" style="width:100%; border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
  </a>
  <a href="assets/c.png" target="_blank">
    <img src="assets/c.png" alt="Tyre Analysis" loading="lazy" style="width:100%; border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
  </a>
  <a href="assets/d.png" target="_blank">
    <img src="assets/d.png" alt="Model Metrics" loading="lazy" style="width:100%; border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
  </a>
  <a href="assets/e.png" target="_blank">
    <img src="assets/e.png" alt="Simulation Interface" loading="lazy" style="width:100%; border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,0.1);">
  </a>
</div>

<details>

</details>

## ✅ Core Advantages
- Real-time strategy simulation
- Uses actual F1 historical data
- Interactive parameter tuning
- Open-source and customizable

## ⚠️ Current Limitations
- Limited to 2022-2023 season data
- Simplified safety car modeling
- Basic tyre degradation assumptions
- Fixed pit stop duration (25s)

## 🔮 Future Development
- [ ] Real-time weather API integration
- [ ] Dynamic safety car probability
- [ ] 2024 season data support
- [ ] Driver performance profiles
- [ ] Multi-team strategy comparison

## 🤝 Contribution Guidelines
Contributions welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
Distributed under MIT License. See `LICENSE` for more information.

## ✉️ Contact
For questions/suggestions:  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-YourProfile-blue)](https://linkedin.com/in/YashVaman)
