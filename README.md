# Black Carbon proxy

Author: Bertille Temple  
Last update: August 22, 2023  
Research group: Statistical Analysis of Networks and Systems (SANS)  
Department: Computers Architecture Department (DAC)  
Institution: Polytechnic University of Catalonia (UPC)  

---

This project was made to predict Black Carbon(BC) concentration in Delhi. It is the result of a collaboration with Pratima Gupta, Ajit Ahlawat, Mar Viana and cie and the research group SANS. 

* ``data`` folder contains the BC and the Relative Humidity(RH) data from Delhi, and the Solar Radiation(SR) data from Agra, 
* ``img`` folder contains images of seasons spliting, true and predicted values among time, true VS predicted values, 
* ``src`` folder contains all the scripts: 
    * main.py is the main script. It has 2 parts: first, predictions are made on the whole dataset, then predictions are made by season. 
    * parameters.py contains the parameters to be selected before running the main script, 
    * predict_BC_class.py contains a big class with all useful functions. These functions are called from main.py. 
    * predict_BC.py contains one function aimed at training and testing a specific ML algorithm to predict BC. This function is called from main.py. 


# Step by step
0) Optional. To avoid dependencies conflict, create a virtual environment with conda for example, activate it, and run ``pip install -r requirements.txt`` from the BC_proxy_Delhy folder. 
1) go to the src folder. A combination of parameters which gives good predictions is set by default in parameters.py. You can update them with a code editor of your choice in the first part of the script, called "DEFINING PARAMETERS". To do so, you will need to ask the questions:  
    * Which method should be used to predict Black Carbon:  Neural Network, Support Vector Regression or Random Forest ?
    * Which score should be used to tune the hyper-parameters and or to asses the quality of the predictions: Mean Squared Error or Mean Absolute Error ?
    * Should the original Relative Humidity measures be used ?
    * Should the Solar Radiation be a feature ? 
    * Should the Relative Humidity be imputed with the new measures obtained from "Relative humidity data_Delhi 2018-2019" file, when they are missing in the original file ? 
    * Should the hyper-parameters be tuned ? 
    * Should images of the results be saved ?
    * Should the whole training set be standardize ?  

    The corresponding parameters are listed below: 

        * method: 'NN'/'SVR'/'RF'
        * scoring: 'neg_root_mean_squared_error'/'neg_mean_absolute_error'
        * RH_included: True/False
        * SR_included: True/False
        * RH_imputed: True/False
        * tune_hyperparameters: True/False
        * save_images: True/False
        * std_all_training: True/False
    Be careful, std_all_training should be set to False, because cross validation is used. 
    

2) in the src folder, run the main.py script with python with the method of your choice. For example, you can run the command ``python3 main.py`` from your terminal. 
