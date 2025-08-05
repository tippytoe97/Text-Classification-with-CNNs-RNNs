# Text Classification with CNNs, RNNs, and Word2Vec

This project explores different deep learning models for classifying text, including **Convolutional Neural Networks (CNNs)**, **Recurrent Neural Networks (RNNs)**, and **Word2Vec embeddings**. It was built using **PyTorch** and applied to real-world datasets for tasks like clickbait detection and article topic classification.

---

## Datasets Used

- **Clickbait Dataset**  
  A collection of news headlines labeled as clickbait or not.

- **Web of Science Dataset (subset)**  
  Scientific article titles and abstracts, labeled by research domain.

---

## Models and Techniques

- **CNN and RNN Classifiers**  
  Used to capture spatial (CNN) and sequential (RNN) patterns in text data.

- **Word2Vec Embeddings**  
  Explored both CBOW and Skip-gram methods. Trained on a small custom dataset and used pre-trained vectors to improve performance.

- **Custom Training Loop (Bonus)**  
  Built a two-layer neural network and wrote a full training loop from scratch for deeper understanding.

---

## Project Structure
├── cnn_model.py # CNN-based classifier
├── rnn_model.py # RNN-based classifier
├── word2vec_utils.py # Word2Vec training and integration
├── README.md # This file

##How to Run:
1. Install dependencies
pip install -r requirements.txt

2. Train the model
You can run either of the models using:
python train.py --model cnn   # For CNN
python train.py --model rnn   # For RNN
(Modify other training options in the script as needed.)

## Results:
Model	Dataset	Accuracy
CNN	Clickbait	~96%
RNN	Web of Science	~94%

(Note: Results may vary based on embedding choice and model tuning.)

## What I Learned:
How CNNs and RNNs process text differently
How to use Word2Vec for feature representation
How to implement and evaluate text classifiers in PyTorch
How embeddings and architecture choices affect performance
