# Progetto Deep Learning

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependences](#dependences)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Mediapipe](#mediapipe)
- [Results](#results)

## Introduction

L'obiettivo del progetto è implementare un sistema in grado di rilevare e classificare le gestualità del linguaggio dei segni a partire da una fonte video o stream. Il sistema dovrà essere in grado di identificare i gesti e associarli alla corrispondente classe di appartenenza. Per questo progetto, ho selezionato un gruppo ristretto di parole, comprensivo di lettere, numeri e termini di uso frequente. L'obiettivo principale è addestrare e confrontare diversi modelli di deep learning, utilizzando architetture studiate durante il corso, al fine di valutarne le performance e comprendere meglio le differenze tra di esse.

## Project Structure

AslDetection/<br>
├── backups/<br>
├── logs/<br>
│   ├── cnn/<br>
│   ├── rnn_gru/<br>
│   ├── rnn_lstm/<br>
├── models/<br>
│   ├── cnn/<br>
│   ├── rnn_gru/<br>
│   ├── rnn_lstm/<br>
├── temp/<br>
│   ├── X.npz<br>
│   ├── y.npz<br>
│   ├── .gitignore<br>
│   ├── collect_keypoints_1.ipynb<br>
│   ├── collect_keypoints_2.ipynb<br>
│   ├── collect_keypoints_3.ipynb<br>
│   ├── config_loader.py<br>
│   ├── constants.py<br>
│   ├── custom_data_generator.py<br>
│   ├── dataset_features.ipynb<br>
│   ├── detect_keypoints.py<br>
│   ├── draw_acc_loss.ipynb<br>
│   ├── draw_model.ipynb<br>
│   ├── globali.py<br>
│   ├── requirements.txt<br>
│   ├── settings.py<br>
│   ├── settings.yaml<br>
│   ├── train_test_1.ipynb<br>
│   ├── train_test_2.ipynb<br>
│   ├── train_test_colab.ipynb<br>
│   ├── WLASL_v0.3.json<br>
│   ├── youtube_asl_video_ids.txt<br>


## Dependences

Le dipendenze da installare sono contenute nel file "requirements.txt". Eseguire il comando:

```bash
pip install -r requirements.txt
```


## Dataset

Il dataset contenente i file .npy dei landmarks estratti con Mediapipe è scaricabile a questo link: <a href="https://drive.google.com/drive/folders/1H2aMWnAQuOfzBfWmw3EBKS3-IQIwGTRB"></a>


## Model Architecture



## Mediapipe

![image](https://github.com/user-attachments/assets/9162f3b1-d30b-4bd9-ad84-ea45c11dfea2)

Mediapipe è un framework open-source di Google che fornisce una serie di moduli predefiniti per il rilevamento di parti del corpo, come mani, braccia,  gambe, busto, testa e volto. Nel caso specifico del progetto è stata impiegata la funzione Holistic per il rilevamento dei landmarks di interesse.


<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/31e57f69-0ab4-4341-bced-7ddd9b3392cf" alt="Example Image" width="400"/>
</div>


## Results

Accuracy
  - CNN: 0.93
  - RNN con LSTM: 0.86
  - RNN con GRU: 0.86



