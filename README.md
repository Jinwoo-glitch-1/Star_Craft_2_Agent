<div align="center">

# 🌌 StarCraft II AI Agent Pipeline

**A high-performance, robust data extraction and preprocessing pipeline bridging the StarCraft II API with Deep Learning architectures.**

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Ready](https://img.shields.io/badge/PyTorch-Ready-%23EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![API](https://img.shields.io/badge/API-python--sc2-purple.svg?style=flat-square)](https://github.com/BurnySc2/python-sc2)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

</div>

<br />

## 📖 Table of Contents
- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Running the Bot](#running-the-bot)
  - [Integrating the Pipeline](#integrating-the-pipeline-for-ml)
- [Roadmap](#-roadmap)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 About The Project

This project provides a foundational Python-based AI agent for StarCraft II. Rather than relying on hardcoded rule-based scripts, this repository focuses on building a **robust data pipeline**. It extracts real-time game state information, engineers advanced time-series features, and normalizes them into clean, standardized tensors.

This environment is purpose-built to feed directly into **Reinforcement Learning (RL)** or **Deep Learning (LSTM)** models via PyTorch, effectively translating complex SC2 macro-mechanics into machine-readable mathematical formats.

---

## ✨ Key Features

- 🤖 **Automated Game Integration:** Seamlessly connects and runs a basic bot directly against the standard StarCraft II in-game AI.
- 📊 **Real-Time State Extraction:** Dynamically pulls critical macroscopic game metrics (Minerals, Vespene Gas, Supply limits, Worker allocations) at optimized step intervals.
- ⚙️ **Advanced Feature Engineering:** Computes complex derived metrics on the fly:
  - **Collection Rate:** Calculates resource gathering velocity (resources per minute).
  - **Human/AI Spending Rate:** Detects resource drops to model spending behaviors and army production pacing.
  - **Saturation Ratio:** Computes worker distribution efficiency across active townhalls.
- 🧮 **Data Normalization:** Implements robust Min-Max scaling tailored to SC2 bounds, ensuring all features are strictly within a `[0, 1]` range (with overflow clipping) for optimal neural network gradient flow.
- 🔥 **PyTorch Ready:** Outputs fully scaled, ready-to-use `float32` NumPy tensors natively compatible with PyTorch modules.

---

## 🏗 System Architecture

```text
Star_Craft_2_Agent/
├── main.py                   # Main entry point to initialize the bot and environment
├── requirements.txt          # Project dependencies
└── utils/
    ├── game_reader.py        # [Step 1 & 2] Extracts raw state & calculates derived metrics
    ├── feature_scaler.py     # [Step 3] Custom Min-Max scaler tuned for SC2 macro bounds
    └── data_pipeline.py      # [Step 4] Pipeline orchestrator outputting final AI tensors
```

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### Prerequisites

- **Python:** Version 3.7 or higher.
- **StarCraft II:** A local installation of [StarCraft II](https://starcraft2.blizzard.com/).
- **Maps:** Ensure the official StarCraft II Map Packs are installed (specifically, the `Simple64` map must be available in your `Maps` directory).

### Installation

1. Clone the repository:
```bash
   git clone https://github.com/your-username/Star_Craft_2_Agent.git
   cd Star_Craft_2_Agent
```

2. Create a virtual environment (Recommended):
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
   pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Bot

To start a local game with the bot (currently configured to play as Protoss against a Very Easy Terran AI), execute the main script:

```bash
python main.py
```

### Integrating the Pipeline for ML

If you are expanding the `MinimalBot` in `main.py` to train an ML/RL model, you can seamlessly plug the pipeline into your bot's `on_step` loop:

```python
import torch
from utils.game_reader import StateExtractor
from utils.feature_scaler import FeatureScaler
from utils.data_pipeline import get_final_ai_input

class MyRLBot(BotAI):
    def __init__(self):
        super().__init__()
        self.extractor = StateExtractor()
        self.scaler = FeatureScaler()
        # self.model = YourPyTorchModel()

    async def on_step(self, iteration: int):
        # Throttle data extraction to reduce overhead (e.g., every 10 steps)
        if iteration % 10 == 0:
            # 1. Get the scaled tensor
            ai_input_array = get_final_ai_input(self, self.extractor, self.scaler)

            # 2. Convert to PyTorch Tensor
            input_tensor = torch.tensor(ai_input_array, dtype=torch.float32)

            # 3. Feed forward to your model
            # prediction = self.model(input_tensor)
```

---

## 🗺️ Roadmap

- [x] Build core state extraction class (`StateExtractor`).
- [x] Implement derived feature logic (Collection rate, Spending rate, Saturation).
- [x] Create custom Min-Max scaler tailored for SC2 metrics.
- [x] Wrap modules into a single execution pipeline.
- [ ] Integrate the data pipeline natively into a PyTorch training loop (LSTM / Fuzzy Logic Controller).
- [ ] Expand extracted features to include specific unit counts, building status, and spatial map vision grids.
- [ ] Implement an Action-Space translator (mapping model tensor outputs back to SC2 game commands).

---

## 🤝 Acknowledgments

- Built utilizing the highly capable [python-sc2 / burnysc2](https://github.com/BurnySc2/python-sc2) wrapper.
- Inspired by AI advancements in real-time strategy environments.

---

<div align="center">
  <i>Built with ❤️ for AI & RTS Enthusiasts.</i>
</div>
