# Pytorch-RNN-create-Q-A-Syste-
a simple RNN to predict answers based on input questions using a custom QA dataset. It tokenizes text, builds a vocabulary, converts words to indices, and feeds them into an embedding + RNN + linear layer. The model is trained with CrossEntropyLoss, and a predict function generates answers using softmax probabilities.
