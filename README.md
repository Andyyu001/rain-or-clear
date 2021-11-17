# rain-or-clear

We aim to train a network for rain/clear image classifition. 
The network is constructed based on the inceptionnet v4. 

## Implementation
python3 
pytorch >= 1.2
NVIDIA GPU + CUDA

## Data
.  
|-- data  
`-- |-- train  
`-- |-- |-- rain  
`-- |-- |-- norain  
`-- |-- test  
`-- |-- |-- ...  

## How to Train
* File 'gen_label_csv.py': get the label file of data. You should change the data path.
* File  'main_dif_rain.py': main file for training    
Open the 'main_dif_rain.py', you can change the hype-parameters in the file.
