# Machine Learning Methods for Detecting Signs of Disease from an ECG

Implementation of experiments described in undergraduated thesis conducted by Andrii Vandzhura and supervised by Anton Popov

## Abstract

Deep learning has shown strong potential in automating ECG analysis to support
clinical decision-making. Most approaches to this problem use either the time domain
or time-frequency domain representation of signals and focuse narrowly on detecting
one specific class of diseases. This study compares deep learning approaches to detect
multiple disease classes from ECG, analyzing the effects of different signal prepro-
cessing and lead reduction on model performance in both time and time-frequency
domains. In the time domain the comparison was conducted both on raw and filtered
ECG signals and on ECGs with full and reduced set of leads, which were evaluated
with LSTM, GRU, and CNN-GRU models. In the time-frequency domain, the per-
formance of ResNet models of different depth trained on spectrograms (STFT) and
scalograms (CWT) were compared. The study showed that signal filtering improved
LSTM performance, but had limited or negative effects on other models. Deeper
ResNets trained on scalograms showed stable or improved secondary metrics, although
spectrogram-based ResNet18 achieved the highest primary metric. The reduction of
ECG leads did not significantly affect prediction quality of models.

## Data

For research we used the [PTB-XL database](https://physionet.org/content/ptb-xl/1.0.3/) collected by Wagner et al. stored in `ptbxl` directory. For the project the data has been further processed and saved locally in `data` directory. Note, that instead of uploading the `ptbxl` and `data` directory with training, validation and testing data to the GitHub, we provide the hyperlink to it on Google Drive due to their large size.

## Repository Organization

The repository structure can be summarized in the following scheme:

```
├── data
|   ├── labels.zip
|   ├── scaleograms.zip
|   ├── signals100.zip
|   ├── signals100filtered.zip
|   └── spectrograms.zip
├── models
|   ├── filtered
|   ├── raw
|   ├── scaleograms
|   └── spectrograms
├── training
|   ├── filtered
|   ├── raw
|   ├── scaleograms
|   └── spectrograms
├── testing
|   ├── filtered
|   ├── raw
|   ├── scaleograms
|   └── spectrograms
├── visualizations  
|   ├── training_validation_plots
|   └── testing_plots
├── ptbxl  
|   ├── records100
|   └── records500
├── architectures_signal.py
├── architectures_spectral.py
├── datasets.py
├── trainer_signals.ipynb
├── trainer_spectral.ipynb
└── visualization.ipynb
```

Here we provide the description of directories that are used in the project.

|Directory|Description|
|---|---|
|[ptbxl](https://drive.google.com/drive/folders/1ouuRNhCcREcsva9WhLlBOuJ1Fht6rCqY?usp=drive_link)|Directory contains PTB-XL database.|
|[data](https://drive.google.com/drive/folders/1H5VFBgcAZ79htBH0j-kLH3xvk6MBeO5l?usp=drive_link)|Directory contains labels, raw and preprocessed ECG signals used for models training, validation and testing.|
|models|Directory contains models trained for experiments on raw and filtered ECG signals and their spectral representations in spectrograms and scaleograms in correponding directories.|
|training|Directory contains metrics recorded for each model throughout its training and validation processes.|
|testing|Directory contains metrics recorded for each model throughout its testing process.|
|visualizations|Directory contains visualizations for each model training, validation and testing processes in corresponding directories.|

Here we provide the description of files that are used in the project.

|File|Description|
|---|---|
|labels.zip|Archive with files with labels for suggested disease classes.|
|scaleograms.zip|Archive with scaleograms (256×256).|
|signals100.zip|Archive with signals sampled at 100Hz.|
|signals100filtered.zip|Archive with signals sampled at 100Hz and filtered.|
|spectrograms.zip|Archive with scaleograms (256×256).|
|architectures_signal.py|File contains implementations of models that detect diseases on signals represented in time domain. Class `CustomClassifierSignal` implements the wrapper around the model, which specifies metrics that measure model  performance and training, validation and testing cycles. The classes `LSTM_Classifier`, `GRU_Classifier` and `CNN_GRU_Classifier` implement discussed LSTM, GRU and CNN-GRU architectures.|
|architectures_spectral.py|File contains implementations of models that detect diseases on signals represented in time-frequency domain. Class `CustomClassifierSpectral` implements the wrapper around the model, which specifies metrics that measure model  performance and training, validation and testing cycles. The class `ResNet_Classifier` implements discussed ResNets architectures.|
|datasets.py|File contains the classes `SignalDataset` and `SpectralDataset` used for creation of datasets for signals represented in time and time-frequency domains correspondingly.|
|trainer_signals.ipynb|File contains the code used to train models on signals in time domain representation. The code monitors models training, validation and testing process in tensorboard and saves model and its training, validation and testing information locally in `models`, `train` and `testing` directory correspondingly.|
|trainer_spectral.ipynb|File contains the code used to train models on signals in time-frequency domain representation. The code monitors models training, validation and testing process in tensorboard and saves model and its training, validation and testing information locally in `models`, `train` and `testing` directory correspondingly.|
|visualization.ipynb|File contains the code used to produce visualizations for the models training, validation and testing processes. To make visualizations, the metrics info in `training` and `testing` directory was used.|

## Set up

Most of the project code was developed in Google Colab. It is highly recommended to download the repository to your Google Drive and run the notebooks directly in Colab.

## Results

| Metric      | LSTM Raw | LSTM Filtered | GRU Raw    | GRU Filtered | CNN-GRU Raw | CNN-GRU Filtered |
|-------------|----------|---------------|------------|--------------|-------------|------------------|
| Loss        | 0.3358   | **0.2926**    | **0.2677** | 0.2730       | 0.2752      | <b>0.2719</b>   |
| AUC         | 0.8546   | <b>0.9010</b>        | <b>0.9229</b>     | 0.9205       | 0.9220      | <span style="color:red"><b>0.9237</b></span>   |
| F<sub>1</sub> score   | 0.5108   | <b>0.6588</b>        | <span style="color:red"><b>0.6989</b></span>     | 0.6774       | 0.6811      | <b>0.6830</b>           |
| F<sub>β</sub> score   | 0.5825   | <b>0.7372</b>        | <span style="color:red"><b>0.7810</b></span>     | 0.7544       | <b>0.7707 </b>     | 0.7640           |
| Precision   | 0.4977   | <b>0.5651</b>        | <b>0.5954</b>     | 0.5911       | 0.5753      | <span style="color:red"><b>0.5967</b></span>   |
| Recall      | 0.6444   | <b>0.8047 <b>       | <b>0.8479</b>     | 0.8229       | <span style="color:red"><b>0.8481</b></span>      | 0.8379           |
| Accuracy    | 0.8204   | <b>0.8419</b>       | <span style="color:red"><b>0.8581</b></span>     | 0.8558       | 0.8461      | <b>0.8522</b>           |
| NPV         | 0.9377   | <b>0.9495</b>       | <b>0.9615</b>     | 0.9589       | <span style="color:red"><b>0.9627</b></span>      | 0.9620           |
| Specificity | 0.8088   | <b>0.8294</b>        | 0.8375     | <span style="color:red"><b>0.8394</b></span>   | 0.8197      | <b>0.8314</b>          |

Through experiments, we have shown that ECG signal filtering has significantly improved the performance of the LSTM model across all metrics. However, for the GRU
and CNN-GRU models, the results are mixed. In particular, GRU produces better results on raw signals, while using filtered signals with CNN-GRU improves primary AUC and secondary F1 score.

| Metric      | LSTM 8-lead | LSTM 12-lead | GRU 8-lead | GRU 12-lead | CNN-GRU 8-lead | CNN-GRU 12-lead |
|-------------|-------------|--------------|------------|-------------|----------------|-----------------|
| Loss        | **0.2830**  | 0.2926       | **0.2696** | 0.2730      | **0.2693**     | 0.2719  |
| AUC         | <b>0.9069</b>  | 0.9010       | 0.9204     | <b>0.9205</b>      | 0.9197         | <span style="color:red"><b>0.9237</b></span>  |
| F<sub>1</sub> score   | **0.6752**  | 0.6588       | <b>0.6774</b> | <b>0.6774</b>      | 0.6766         | <span style="color:red"><b>0.6830</b></span>  |
| F<sub>β</sub> score   | **0.7477**  | 0.7372       | 0.7494 | <b>0.7544</b>      | <span style="color:red"><b>0.7829</b></span>  | 0.7640          |
| Precision   | <b>0.5879</b>  | 0.5651       | <b>0.5944</b> | 0.5911      | 0.5529         | <span style="color:red"><b>0.5967</b></span>  |
| Recall      | <b>0.8102</b>      | 0.8047       | 0.8124     | <b>0.8229</b>      | <span style="color:red"><b>0.8755</b></span>  | 0.8379          |
| Accuracy    | 0.8416      | <b>0.8419</b>       | <span style="color:red"><b>0.8561</b></span> | 0.8558      | 0.8385         | <b>0.8522</b>          |
| NPV         | <b>0.9528</b>  | 0.9495       | 0.9560     | <b>0.9589</b>      | <span style="color:red"><b>0.9663</b></span>  | 0.9620          |
| Specificity | 0.8232      | <b>0.8294</b>       | <span style="color:red"><b>0.8414</b></span> | 0.8394      | 0.8081         | <b>0.8314</b>          |

Reducing 12-lead ECG signal to 8-lead generally maintains or slightly improves model performance, especially for LSTM, suggesting that reduced-lead ECGs can still be effective for pathology detection. This observation can be used to decrease the amount of training data for deep learning models without a significant loss in disease detection performance.

| Metric      | ResNet18 Spec. | ResNet18 Scale. | ResNet34 Spec. | ResNet34 Scale. | ResNet50 Spec. | ResNet50 Scale. |
|-------------|----------------|-----------------|----------------|-----------------|----------------|-----------------|
| Loss        | **0.3155**     | 0.3202          | 0.3230         | **0.3205**      | 0.3376         | <b>0.3186</b>  |
| AUC         | <span style="color:red"><b>0.8718</b></span> | 0.8624          | 0.8574         | **0.8615**          | 0.8430         | **0.8662**          |
| F<sub>1</sub> score   | **0.6319**            | 0.6313          | 0.6068         | <span style="color:red"><b>0.6338</b></span>      | 0.5572         | <b>0.6306</b>  |
| F<sub>β</sub> score   | 0.6588            | **0.6763**          | 0.6316         | **0.6680**          | 0.5740         | <span style="color:red"><b>0.6854</b></span>  |
| Precision   | <span style="color:red"><b>0.5978</b></span> | 0.5693          | 0.5726         | <b>0.5853</b>     | 0.5428         | <b>0.5578</b>  |
| Recall      | 0.6800            | **0.7104**          | 0.6504         | **0.6935**          | 0.5887         | <span style="color:red"><b>0.7284</b></span>  |
| Accuracy    | <span style="color:red"><b>0.8507</b></span> | 0.8348          | 0.8440         | <b>0.8444</b>      | 0.8311         | **0.8366**          |
| NPV         | <span style="color:red"><b>0.9281</b></span> | 0.9224          | 0.9192         | <b>0.9231</b>      | 0.9081         | **0.9214**          |
| Specificity | **0.8617**            | 0.8441          | <span style="color:red"><b>0.8621</b></span> | 0.8585          | **0.8533**         | 0.8495          |

As the complexity of the ResNet model increases, performance with spectrograms declines, but scaleograms allow for more stable or slightly improved performance, especially in recall and Fβ score, indicating that scalograms are more robust for deep models training.
