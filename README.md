# Petfinder Pawpularity

## 📌 Project Overview  
This project analyzes the **Petfinder Pawpularity Contest**. The dataset consist of pet images and meta data and we need to find the Pawpualrity score(cuteness). In addition we overiew models structure and their strength. In the end, we will propose some improvement to existing models to get better performance.

It demonstrates a **vision transformer based models**, including:  
- **Vision Transformer (Images+metadata)** model  
- **Swin transformer (Image only)** model    
- **Swin tranformer with Images+metadata - Extended PETS-SWINF**    

# Usage
main.ipynb is the main and entry file, the other existing approaches are defined in the "existing_approaches.ipynb" file
   
## 📂 Project Structure 
├── src/  
│ ├── data.py 
│ ├── train.py  
│ ├── models.py 
│ ├── eval.py  
├── utils/ # Helper functions  
│ ├── helpers.py #helper functions for graph, metrics plotting  
├── main.ipynb # Main Entry point of Extended SWIN Tranformer  
├── existing_approaches.ipynb # Existing models Exploration    
├── PDF Report 