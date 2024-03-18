# GREAT-CARE (Alpha-Test) Dashboard

Welcome to GREAT-CARE (Alpha-Test) Dashboard! This dashboard is designed to predict and recommend diagnosis and medication for patients during a visit based on their medical conditions such as symptoms, diagnosis, procedures, and medications. The predictions are generated using a Graph Neural Network (GNN) model. Additionally, the dashboard integrates the CORE AI component (LLM), which analyzes doctors' proposals and facilitates collaborative discussions among medical team members for the final decision on the patient's treatment.

## Prerequisites

Make sure you have the following installed on your machine:

- Python (version 3.9+)
- Pip (version 22.0+)

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/giuseppericcio/MSc-Thesis-Project.git
    ```

2. Navigate to the project directory:

    ```bash
    cd MSc-Thesis-Project
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Dashboard

To run the GREAT-CARE (Alpha-Test) Streamlit dashboard, use the following command:

```bash
streamlit run dashboard.py
```

This will start a development server, and you can view the dashboard in your web browser at http://localhost:8501.
