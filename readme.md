# CoReBM

## Introduction
This repository is divided into three parts: **Comparison**, **Dataset** and **Implementation**. 

**Note**: Detailed introduction can be found in the [Introduction] section of our paper, which is currently under review. Stay tuned!

## General of Packages
/Comparison: Comparative experiment.  
/Dataset: Dataset used in experiment.  
/Implementation: Implementation of the proposed method.

## Getting Started

### Usage Guide
If you want to try out our method, please copy all files located in the `/Implementation` folder and open them in a new project. You can choose to use either the command line or a browser.

### Set up the Environment
Note: This environment is only related to the CoReBM method. We have not provided environment description for the comparative experiment.
1. Python version should be greater than to 3.10.13.

2. Install dependencies:
    ```shell
    pip install -r requirements.txt
    ```
### Dataset
Please place the dataset (four files with the *.jsonl suffix) in the `data/corebm/raw_data` folder. Additionally, due to GitHub's limitations, we have not yet uploaded the dataset to the repository. Please download the dataset here: [https://drive.google.com](https://drive.google.com/file/d/11JfeGkVqb4M3zKSJEWSw3iZ9bFIqWyrJ/view?usp=sharing)

### Run With Command Line

```shell
python main.py --main Evaluate --data_file data/corebm/test.csv --system collaboration --system_config config/systems/collaboration/all_agents.json --task pr --rounds 1
```

### Run With Browser

```shell
streamlit run web.py
```

Visit through `http://localhost:8501/`.

