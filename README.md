# Progetto Deep Learning

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependences](#dependences)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

L'obiettivo del progetto è implementare un sistema in grado di rilevare e classificare le gestualità del linguaggio dei segni a partire da una fonte video o stream. Il sistema dovrà essere in grado di identificare i gesti e associarli alla corrispondente classe di appartenenza. Per questo progetto, ho selezionato un gruppo ristretto di parole, comprensivo di lettere, numeri e termini di uso frequente. L'obiettivo principale è addestrare e confrontare diversi modelli di deep learning, utilizzando architetture studiate durante il corso, al fine di valutarne le performance e comprendere meglio le differenze tra di esse.

## Project Structure

AslDetection/<br>
├── backups/<br>
├── logs/<br>
│   ├── cnn/<br>
│   ├── rnn_gru/<br>
│   ├── rnn_lstm/<br>
├── models/
│   ├── cnn/
│   ├── rnn_gru/
│   ├── rnn_lstm/
├── temp/
│   ├── X.npz
│   ├── y.npz
│   ├── .gitignore
│   ├── collect_keypoints_1.ipynb
│   ├── collect_keypoints_2.ipynb
│   ├── collect_keypoints_3.ipynb
│   ├── config_loader.py
│   ├── constants.py
│   ├── custom_data_generator.py
│   ├── dataset_features.ipynb
│   ├── detect_keypoints.py
│   ├── draw_acc_loss.ipynb
│   ├── draw_model.ipynb
│   ├── globali.py
│   ├── requirements.txt
│   ├── settings.py
│   ├── settings.yaml
│   ├── train_test_1.ipynb
│   ├── train_test_2.ipynb
│   ├── train_test_colab.ipynb
│   ├── WLASL_v0.3.json
│   ├── youtube_asl_video_ids.txt


## Dependences

Le dipendenze da installare sono contenute nel file "requirements.txt". Eseguire il comando:

```bash
pip install -r requirements.txt

