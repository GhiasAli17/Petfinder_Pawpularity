# Petfinder Pawpularity

## 📌 Project Overview  
This project analyzes the **Petfinder Pawpularity Contest**. The dataset consist of pet images and meta data and we need to find the Pawpualrity score(cuteness). In addition we overiew models structure and their strength. In the end, we will propose some improvement to existing models to get better performance.

It demonstrates a **vision transformer based models**, including:  
- **Vision Transformer (Images+metadata)** model  
- **Swin transformer (Image only)** model    
- **Swin tranformer with Images+metadata - Extended PETS-SWINF**    

# Usage
main.ipynb is the main and entry file, where other modules are imported, the flow and visualization can be viewed in the main.ipynb file
   
## 📂 Project Structure 
├── src/  
│ ├── data.py # data loading  
│ ├── train.py # models training  
│ ├── models.py # model's architecture definitions,  
│ ├── eval.py # testing models  
├── utils/ # Helper functions  
│ ├── helpers.py #helper functions for graph, metrics plotting  
├── main.ipynb # Entry point   
├── PDF Report 