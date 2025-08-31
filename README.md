# Speech-Vs-Noise AI Project
This project is my journey to build an AI model that can separate human speech from background noise. 
The idea comes from personal experience with hearing loss in my family, and my hope is to explore how math + technology can make sound clearer for people who need it most. 

Day 1 - Environment Setup
- Installed Python 3.12
- Installed VSCode for coding and Git integration
- Set up Github desktop for version control
- Created a repository and made my first commit

Day 2 - Dataset and First visualization
- Downloaded urbansound8k dataset
- Organized project structure
   - data/speech: placehodler for speech samples
   - data/noise: contains background noise samples
- Created first Python script: day2_visualize.py
   - Loads an audio file with Librosa
   - Visualizes sound as a waveform using Matplotlib
   - Visualized dog bark as a waveform

- Day 2 was about preparing data and making audio visible for the first time.
- Seeing the waveform of a dog bark helped me realize that sound is just numbers.
- Those numbers can be turned into patterns that could be analyzed and that a machine can learn.


Day 3 - Preprocessing and Mel Spectrogram
- Wrote a Python script: 'preprocess.py'
  - Trims leading and trailing silence from audio clips
  - Normalizes loudness so files are on the same scale
  - Cuts or pads every clip to exactly 3 seconds
  - Saves processed clips into 'data/processed/...'
 
- Created 'day3_visualize.py' to plot a mel spectrogram
  - Displays sound as a time frequency heatmap
  - Used on a processed dog bark file
 
- Day 3 was about about preparing consistent inputs for machine learning.
  - I learned that for a model to compare sounds fairly, all audio must be the same length and volume.
  - The mel spectrogram made sound look like a picture, showing how noise looks random while speech forms clearer patterns.

Day 4 - MFCC Feature Extraction 
Wrote a Python Script: 'day4_mfcc_extract.py'
- Extracts Mel-Frequency Cepstral Coefficients (MFCC) from processed 3-second clips
- Saves per file MFCC matrices into 'feature/'
- Builds a combined dataset 'dataset.npz' with compact feature vectors (speech = 1, noise = 0)

Created 'day4_visualize_mfcc.py' to plot MFCC spectrograms
- Displays MFCCs as a heatmap of coefficients over time
- Speech shows smoother, repeating patterns.
- Noise appears scattered and irregular.

Day 4 was about transforming sound into mathematical features that a machines learning model can use.
I learned that MFCCs capture the important structure of sound while ignoring extra noise. 
It reminded me of how my mom's hearing aids emphasizes key frequencies - the same principle I applied today through math. 

Day 5- 

