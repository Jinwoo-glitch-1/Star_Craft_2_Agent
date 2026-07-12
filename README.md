<div align="center">

# 🌌 SC2 Micro-Trainer Assistant
**Your AI Co-Pilot for StarCraft II. You focus on the battles, let the AI handle the economy.**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Powered](https://img.shields.io/badge/PyTorch-Powered-%23EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Fuzzy Logic](https://img.shields.io/badge/Engine-Scikit--Fuzzy-orange.svg?style=flat-square)](#)
[![Status: Production](https://img.shields.io/badge/Status-Production_Ready-success.svg?style=flat-square)](#)

</div>

<br />

## 🎯 What is the SC2 Micro-Trainer?

Playing StarCraft II is incredibly demanding, requiring players to split their attention between commanding armies (Micro) and building their economy (Macro). 

The **SC2 Micro-Trainer Assistant** is a production-ready AI Co-Pilot designed to help you practice your combat skills. Once launched, you and the AI share control of the same base. **You command the army and fight the wars, while the AI intelligently builds workers, manages supply, harvests gas, and expands to new bases in the background.**

It is powered by a hybrid Deep Learning architecture, utilizing an **LSTM Neural Network** to predict starvation risks and a **Fuzzy Logic Controller** to make human-like expansion decisions.

---

## ✨ Key Features

* **🎮 Interactive Match Setup:** A sleek terminal interface allows you to select your race, the opponent's race, AI difficulty, and choose from 19 different competitive maps before the game launches.
* **🧠 Hybrid AI Brain:** * **LSTM Resource Predictor:** Analyzes the last 60 frames of gameplay to predict economic starvation before it happens.
    * **Fuzzy Logic Expansion:** Evaluates mineral saturation, time, and resources to decide the optimal moment to build new bases.
* **🛡️ Auto-Recovery Protocols:**
    * *Worker Retreat:* Automatically pulls workers away from enemy attacks to safe bases.
    * *Emergency Expansion:* If your main base is destroyed, the AI triggers an emergency protocol to rebuild in a hidden location.
    * *Idle Audit:* Constantly audits workers to ensure 0% idle time.
* **⚡ 1-Click Deployment:** No coding required! Comes with an automated `.bat` launcher that creates a secure environment, installs dependencies, and launches the game automatically.
* **🌍 Multi-Race Support:** Play as Terran, Zerg, or Protoss. The AI seamlessly adapts its macro strategy to your chosen race.

---

## 🏗 System Architecture

The assistant is built on a robust, decoupled architecture ensuring high performance during real-time gameplay:

```text
SC2_Micro_Trainer/
├── Play_With_Assistant.bat   # 1-Click Automated Launcher for Gamers
├── main.py                   # Match orchestrator & interactive setup menu
├── requirements.txt          # Python dependencies
├── models/
│   └── resource_predictor.pth # Pre-trained LSTM PyTorch weights
├── ai_models/
│   ├── data_pipeline.py      # Real-time state extraction & tensor normalization
│   ├── fuzzy_controller.py   # Scikit-Fuzzy decision engine
│   └── lstm_model.py         # Neural Network architecture
└── bot/
    └── core_bot.py           # The SC2 Bot integration and macro execution logic
