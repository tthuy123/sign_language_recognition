# sign_language_recognition

## Installation and Usage
1. Install requirement
   ```sh
   pip install -r requirements.txt
   ```
2. Update parameter:
   - Enter your API key in `grammar_correction.py` to acess to LLMs
   - Define and update your webcam index to cv2
   - Define list of action that want to train and recognite
4. Data collection:
   - Run `data_setup` to create data store folder
   - Run `data_collect.py` to start collect data from your webcam.
5. Training model:
   - Run `model.py` to start training model or load model using`.h5` file 
6. Recognition realtime
   - Run 'main.py` to start app.
     + Pess `space` to reset sentence
     + Press `enter` to grammar correction
     + Press `q` to exit

