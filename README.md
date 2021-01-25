# Speech-Enhancement
## Dataset Used
In this project we have used clean voices and some noise files and then we blended both to make our training and testing voices
### The file directory should be this way
![1_IAMXYwz09MQSJdvdy12vDA](https://user-images.githubusercontent.com/63897550/105654522-2d454180-5ee4-11eb-871e-937ffe41f271.png)
##
Clean Voices - clean voices are taken from Librispeech - http://www.openslr.org/12/ which is just day to day talking of people
##
Noise - Noise notes are taken from the official website of columbia University - https://www.ee.columbia.edu/~dpwe/sounds/noise/ and there are 7 files of noise used here - link - https://drive.google.com/drive/folders/1QgLOFAD4yOY6gtmzIN9A2Xdlz2ngcwAh?usp=sharing
##
Blended_note/Training Dataset - we have mixed both the noises and made dataset of nearly 1hr of voicenote mixed with noise which we will use for training here is the link of voice note - https://drive.google.com/file/d/1cfLD7mweR6jwGUlzlJq36DjQBD6PvrmT/view?usp=sharing

## Training Model
### Model Used - Unet
Unet - U-Net is a convolutional neural network that was developed for biomedical image segmentation
### Model Snapshot
for model check this directory - https://github.com/PankajKumarBeniwal/Speech-Enhancement/blob/main/Training/model_unet.py 
there are 10 layers and activation for all the first 9 layers is - "LeakyRelu" and activation at 10th layer is - "tanh" and optimizer used is "Adam"

![m1](https://user-images.githubusercontent.com/63897550/105655637-6da5bf00-5ee6-11eb-872c-9f0af9e27cc3.png)

### Training and Validation Loss
![Figure_1](https://user-images.githubusercontent.com/63897550/105655755-b3628780-5ee6-11eb-8745-ecf48cea30e0.png)

## Functions Used 
Functions used in this model are - https://github.com/PankajKumarBeniwal/Speech-Enhancement/blob/main/Functions_Used/data_tools.py

Functions - 
### 1. audio_to_audio_frame_stack -
This function create an array of all the frames extracted from audio files using sliding window of min_duration = 1.0 sec
### 2. audio_files_to_numpy -
Now this functions converts all this frames into the numpy array form which will be further used to blend both voice_notes and noise_notes
### 3. blend_noise_randomly - 
This functions takes both the numpy array of noise and voice notes and randomly it mixes both arrays and convert it to a new concatinated array
### 4. audio_to_magnitude_db_and_phase -
This function takes an audio and convert into spectrogram,it returns the magnitude in dB and the phase using "Short-time Fourier transform"(stft) method 
### 5. numpy_audio_to_matrix_spectrogram -
This function takes as input a numpy audi of size (nb_frame,frame_length), and return a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size (nb_frame,dim_square_spec,dim_square_spec)
### 6. magnitude_db_and_phase_to_audio -
This function reverts back a spectrogram to audio
### 7. matrix_spectrogram_to_numpy_audio - 
This functions reverts the matrix spectrograms to numpy audio
### 8. scaled_in -
global scaling apply to noisy voice spectrograms (scale between -1 and 1)
### 9. scaled_ou -
global scaling apply to noise models spectrograms (scale between -1 and 1)"
### 10. inv_scaled_in -
inverse global scaling apply to noisy voices spectrograms
### 11. inv_scaled_ou - 
inverse global scaling apply to noise models spectrograms

## Predictions -
Based on above dataset and model the file used here are -
### Testing File - 
blended with noise 

https://drive.google.com/file/d/1SpWnuytdNBH16Jnn0GILsxlKA6dZEyeh/view?usp=sharing
### Noise file -
https://drive.google.com/drive/folders/1QgLOFAD4yOY6gtmzIN9A2Xdlz2ngcwAh?usp=sharing
### Predicted File -
https://drive.google.com/file/d/1-LPrdoIdDauE5d9Z4Ij_6hWJKZ8VJQm5/view?usp=sharing
### Actual file - 
https://drive.google.com/file/d/1e29VSk-rPOYLFL66stGnGDpNhL7MOhEU/view?usp=sharing

### Refrences- 
Jansson, Andreas, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar and Tillman Weyde.Singing Voice Separation with Deep U-Net Convolutional Networks. ISMIR (2017).
[https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf]

Grais, Emad M. and Plumbley, Mark D., Single Channel Audio Source Separation using Convolutional Denoising Autoencoders (2017).
[https://arxiv.org/abs/1703.08019]

Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention — MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham
[https://arxiv.org/abs/1505.04597]

K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015.
[DOI: http://dx.doi.org/10.1145/2733373.2806390]
