# Competitive Matching of Taxis

## Project Overview

This repository contains the code and documentation for a capstone project on optimizing taxi resource allocation in a competitive multi-company environment. The project aims to develop an innovative algorithm that accounts for inter-company competition, enhancing both customer satisfaction and operational efficiency in urban transportation systems.

### Key Features

- Data-driven analysis of taxi markets in New York City and Chicago
- Development of a competitive matching algorithm using game theory and multi-agent reinforcement learning
- Simulation environment for testing and refining the algorithm
- Comprehensive performance evaluation metrics

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Data Analysis](#data-analysis)
4. [Algorithm](#algorithm)
5. [Results](#results)
6. [Acknowledgements](#acknowledgements)

## Installation

```bash
git clone https://github.com/Kalp0811/Competitive-Matching-of-Taxis
cd Competitive-Matching-of-Taxis
pip install -r requirements.txt
```

## Usage

To run the simulation:
```bash
python comp.py
```

## Data Analysis
The project analyzed extensive datasets from New York City and Chicago taxi markets. Key findings include:

- Temporal patterns of taxi trips
- Market share dynamics among competing companies
- Spatial distribution of pickups and dropoffs
- Impact of ride-hailing services on traditional taxis

## Algorithm
The core of the project is a competitive matching algorithm that incorporates:

- Game theory modeling of inter-company competition
- Multi-agent reinforcement learning for adaptive strategies
- Dynamic pricing based on real-time market conditions
- Demand prediction using machine learning models

## Results
Algorithm demonstrated significant improvements in key performance metrics:

- Increased service rates (up to 91.23% for 150,000 requests)
- Reduced average wait times (up to 30.3% reduction)
- Balanced market share distribution among competing companies
- Improved operational efficiency and resource utilization

## Acknowledgements

- Singapore Management University
- Prof. Pradeep Varakantham (Project Supervisor)
- Jiang Hao (PhD student)
- New York City Taxi & Limousine Commission for providing the NYC dataset
- City of Chicago for providing the Chicago Taxi Trips dataset
