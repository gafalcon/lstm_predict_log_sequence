Web Interaction Prediction from Log File

This project contains 3 different modules:

1. Prediction module
Language: Python v3
Libraries:
# Loading data and training/predicting
- pytorch 
- pandas
- numpy
# Visualize results
- matplotlib
- seaborn
- hiddenlayer

Use:
- need to install required libraries first (pip install -r requirements)
- Training LSTM or GRU:
python train.py #Add -h to check options

- Training Seq2Seq model
python train_seq.py #Add -h to check options

2. Server
Language: Python v3
Database: MongoDB
Libraries:
- flask
- flask_socketio
- flask_cors
- requests
- flask_mongoengine

Use: #Needs to have mongo database running on default port 
- need to install required libraries first (pip install -r requirements)
- python server.py

3. Client (predict_client)
Language: Javascript
Libraries
- axios
- chart.js
- react
- react-chartjs-2
- socket.io-client

Use: 
- cd into predict_client folder
- npm i to install libraries
- npm start to execute
