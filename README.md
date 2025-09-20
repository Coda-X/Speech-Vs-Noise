# Speech-Vs-Noise AI Project
This project is my journey to build an AI model that can separate human speech from background noise. 
The idea comes from personal experience with hearing loss in my family, and my hope is to explore how math + technology can make sound clearer for people who need it most. 

**Day 1 - Environment Setup**
- Installed Python 3.12
- Installed VSCode for coding and Git integration
- Set up Github desktop for version control
- Created a repository and made my first commit

**Day 2 - Dataset and First visualization**
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


**Day 3 - Preprocessing and Mel Spectrogram**
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

**Day 4 - MFCC Feature Extraction**
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

**Day 5- First Model Training**
Wrote a Python Script: 'day5_train_model.py'

- Loaded MFCC features from 'features/dataset.npz' (built in Day 4)
- Split data into training and testing sets using a stratified split
- Built two simple machine learning models:
  - Logistic Regression (baseline)
  - Random Forest (alternative)
- Trained the model to classify each clip as:
  - Speech = 1
  - Noise = 0
- Evaluated the results using:
  - Accuracy score
  - Confusion matrix
  - Classification report

The model successfully recognized speech most of the time, but some noises were misclassified as speech.  
This is expected for an early version with limited data.

Day 5 was about taking the first step into actual AI training and seeing my code turn sound into decisions.  
It reminded me of when my mom first started using her hearing aids.  
They weren’t perfect at first and often made mistakes, but every small improvement helped bring clarity to her world.  
Today felt like the same journey, but for my AI model.

**Day 6- Model Improvement and Evaluation**
Wrote Python Scripts:
- 'day6_tune_model.py
  - Performed hyperparameter tuning on both Logistic Regression and Random Forest models
  - Used StratifiedKFold cross-validation to find the best settings for each model
  - Saved the best-performing model to 'features/models/day6_best.joblib'
  - Avoided multiprocessing issues by running safely in single-thread mode

- 'day6_augment.py
  - Added data augmentation to create more training variety
  - Techniques included:
    - Time stretching (slightly faster or slower)
    - Pitch shifting (up and down)
    - Adding Gaussian noise
  - Generated augmented files and saved them into 'data/processed/aug'

- 'day6_build_dataset.py'
  - Rebuilt the dataset to include both original and augmented files
  - Produced updated 'features/dataset.npz' with expanded data

- 'day6_evaluate.py'
  - Evaluated the tuned model with test data
  - Displayed accuracy, classification reports, and confusion matrix
  - Added optional validation curve to visualize model performance

- Augmentation increased the variety of training clips, making the model more robust.
- Hyperparameter tuning improved accuracy compared to the baseline from Day 5.
- The confusion matrix showed better handling of tricky cases, though some noises were still misclassified.

Day 6 was about refining my AI model and moving from a basic first attempt to a smarter, more resilient version.  
The process felt like tuning my mom’s hearing aids:
- Small adjustments brought clearer, more reliable sound.  
- Just like how her device improves over time, my model became sharper and more accurate.

This reminded me that improvement isn’t instant — it takes patience and iteration.

**Day 7- Visualizing my Model's Learning**
Wrote Python Scripts:
- 'day7_learning_curves.py'
  - Plots learning curve to show how accuracy changes with different training sizes
  - Plots validation curve to show performance vs model complexity (C value for Logistic Regression)
  - Saved plots to 'features/plots/day7_learning_curve.png' and 'day7_validation_curve.png'

- 'day7_confusion_matrix.py'
  - Generates a clean, final confusion matrix for visualization
  - Saves output to 'feautres/plots/day7_confusion_matrix.png'

- 'day7_predict.py'
  - Tests the model with **real-world audio clips**:
    - 'hello.wav' → expected Speech
    - 'fan.wav' → expected Noise
  - Loads trained model ('day6_best.joblib')
  - Prints predictions and confidence scores

- Learning curve showed accuracy improving as more training data was added.
- Validation curve helped visualize the balance between underfitting and overfitting.
- Confusion matrix displayed strong performance for speech, with some remaining confusion in complex noise.
- The real-time test correctly identified:
  - My recorded "Hello" as Speech
  - My fan recording as Noise

Day 7 made my AI feel real and tangible for the first time.  
Seeing numbers on a chart and hearing predictions happen live gave me confidence that the project was truly working.

It reminded me of when my mom first heard clearer sounds through her hearing aids: a visual and auditory confirmation that progress was real.

**Day 8 - Finding Weaknesses**
Wrote Python Scripts:
- 'day8_analyze_errors.py'
  - Evaluates the trained model and finds misclassified samples
  - Generates:
    - 'day8_predictions.csv' → all predictions and probabilities
    - 'day8_misclassified.csv' → only wrong predictions
  - Copies misclassified files into 'data/failues' for easy review
  - Saves a confusion matrix plot to 'features/plots/day8_confusion_matrix.png'

- 'day8_compare_waveforms.py'
  - Visualizes why mistakes happened:
    - Plots waveform and mel spectrogram for a misclassified clip
    - Side-by-side comparison with a correct example
  - Highlights how certain noises resemble speech

- 'day8_summary.py' (not in video)
  - Prints a summary of top mistakes
  - Helps identify which sounds were most confusing for the model


- Identified several challenging noise types that were mistaken for speech.
- Collected failed samples for review and further dataset improvement.
- Visualizations revealed overlapping frequency patterns between tricky noises and real speech.

Day 8 taught me that mistakes are valuable data points.  
By seeing exactly where my AI failed, I can better understand how to improve it.

It reminded me of my mom’s experience with her hearing aids:
- At first, they made errors, mixing up voices and background noise.
- Over time, with adjustments and learning, clarity improved.

Failures aren’t the end: they’re the first step toward growth.

**Day 9 - CNN Upgrade**

Wrote Python Scripts:
- 'day9_build_cnn_dataset.py'
  - Converts processed audio clips into **log-mel spectrograms** for CNN input
  - Standardizes each spectrogram to shape '(64, 128, 1)'
  - Saves the dataset to 'features/cnn_dataset.npz'
  - Exports example spectrogram images to 'features/plots' for review

- 'day9_cnn_train.py'
  - Builds and trains a Convolutional Neural Network using TensorFlow/Keras
  - Adds early stopping and model checkpointing to prevent overfitting
  - Generates training curves for **accuracy** and **loss**
  - Saves the best trained model to 'features/models/day9_cnn_best.h5'
  - Creates confusion matrix visualization and saves to 'features/plots/day9_cnn_confusion.png'

- 'day9_compare_old_vs_cnn.py'
  - Compares Day 6 model accuracy vs the new CNN on the **same validation set**
  - Generates a bar chart to visually show improvement
  - Saves comparison chart to 'features/plots/day9_model_compare.png'


- CNN achieved **significantly higher accuracy** than the Day 6 model, especially on complex, overlapping noise clips.
- Log-mel spectrograms allowed the CNN to “see” audio patterns like images.
- The comparison chart clearly showed CNN > Old Model performance.

Day 9 was a major step forward in my project.  
At first, CNNs felt intimidating and complicated, but once I broke the problem into smaller steps, I realized they’re just another tool for learning patterns.

Seeing the CNN outperform my original model was exciting.  
It felt like a glimpse into how deep learning can create real-world solutions.

This reminded me of my mom’s hearing aids:
- With better technology and tuning, clarity improves step by step.
- Just like her devices, my model became more accurate and reliable.

Day 9 taught me that embracing harder challenges leads to growth, both for me and my AI.


