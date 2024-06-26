# Project Title: Deep Learning Trading Bot

## Project Overview
This project implements a reinforcement learning-based trading bot using PyTorch and Backtrader. The bot is designed to learn optimal trading strategies in financial markets by interacting with historical price data.

## Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jikalvin/currency-trading-integrating-deep-q-networks.git
   cd deep-learning-trading-bot
   ```

2. **Set Up a Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv venv
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On MacOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

1. **Activate the Virtual Environment** (if you set one up)
   ```bash
   # On Windows:
   venv\Scripts\activate
   # On MacOS/Linux:
   source venv/bin/activate
   ```

2. **Run the Main Script**
   ```bash
   python main.py
   ```

## Project Structure
- `main.py`: Entry point of the trading bot. Configures and runs the trading strategies.
- `agent.py`: Contains the reinforcement learning agent.
- `strategies/`: Directory containing different trading strategies implemented as Python modules.
- `data/`: Directory for storing historical price data files.

## Additional Notes
- Ensure you have a stable internet connection when downloading financial data using yfinance.
- Adjust the trading parameters in `main.py` based on your specific trading preferences and risk tolerance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
