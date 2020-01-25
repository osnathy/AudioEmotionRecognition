# AudioEmotionRecognition" 

References

    [1] K. Han, D. Yu, and I. Tashev, "Speech emotion recognition using deep neural network and extreme learning machine," in Proc. Interspeech, 2014.

    https://www.microsoft.com/en-us/research/publication/speech-emotion-recognition-using-deep-neural-network-and-extreme-learning-machine/



## Setup:

 1. Python3.7
 
        Note : The code was developed in MAC Catalina OS
 
 2. Data 
 
        We used the IEMOCAP database 
    
        Inorder to use it please ask permissions from :
    
         https://sail.usc.edu/iemocap/
    
 
## How it works ?
 
### 1. Training Phase

 The Project training has 5 main stages as described below:

    1. Data preparation
        
        Run  execute_data_preparation_step.py
        
        Output : utterance_information_file_name.pickle
        
        
    2. Calculate segment level features
    
        Run execute_segment_level_featire_extraction.py
        
        Output : segment_level_features.pickle
        
    3.  Train DNN model
        
        Run execute_dnn.py
        
        Output : dnn_model.json , dnn_model_weights.h5
        
    4. Calculate statistical features
    
        Run execute_statistical_calculation.py
        
        Output : NOT READY YET
        
    5. Train Classifier model (in our implementation we train Random Forest Classifier instead of Extream Machine)
    
        Run execute_classifier.py
        
        Output : random_forest_model.pickel
        
             
### 2. Prediction Phase
    1. Run the flask_api.py
    
    2. From Postman execute POST request to:
    
        http://localhost:8085/predictor
        
        with the jason body:
        '''
        {
          "file_name":"WavFileName.wav"
        }
        ''
        
        Note: Please save the wav file under the following directory :
                \Predict\flask\examples
                
                
     3. The response body will be series of class prediction for each audio segment 
        example : [ 5.0,2.0,5.0,5.0, .... ,2.0]
        
        Note: For next step we will add a summary calculation :)
                
    
        
 