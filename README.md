# Shinkansen-Travel-Experience
The goal of the problem is to predict whether a passenger was satisfied or not considering his/her overall experience of traveling on the Shinkansen Bullet Train.
# Shinkansen Bullet Train Passenger Experience Analysis

## Project Overview

This project analyzes passenger experience data from Japan's Shinkansen Bullet Train, aiming to understand the factors contributing to passengers' overall travel satisfaction. Using machine learning techniques, we will identify the parameters that significantly impact positive or negative feedback from passengers.

## Problem Statement

The Shinkansen Bullet Train is renowned for its punctuality and efficiency. However, to further enhance passenger satisfaction, we seek to understand which specific aspects of the travel experience influence passengers’ overall satisfaction. This analysis will focus on determining the relative importance of different parameters contributing to a positive or negative travel experience.

## Dataset Description

The analysis is based on two primary datasets:

1. **Travel Data** (`Traveldata_train.csv`): Contains information on on-time performance and various passenger details.
2. **Survey Data** (`Surveydata_train.csv`): Provides feedback from passengers on various parameters related to their travel experience, including their overall satisfaction level captured under the variable `Overall_Experience`.

Both files contain records from a sample of individuals who traveled on the Shinkansen Bullet Train and completed the feedback survey.

## Objective

The objective of this project is to identify the key factors influencing positive passenger feedback. Specifically, we aim to:

- Determine which parameters most significantly contribute to overall passenger satisfaction (`Overall_Experience`).
- Understand the relationship between on-time performance and passenger satisfaction.
- Build a predictive model to analyze the importance of each parameter in shaping overall experience.

## Project Structure

- **Data**: Contains `Traveldata_train.csv` and `Surveydata_train.csv`.
- **Notebooks**: Jupyter notebooks for data exploration, model training, and analysis.
- **Models**: Contains trained models and scripts for feature importance analysis.
- **Reports**: Documentation and visualizations based on model outcomes.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Sourena-Mohit/Shinkansen-Travel-Experience-For-Hackathons.git
   cd Shinkansen-Travel-Experience
