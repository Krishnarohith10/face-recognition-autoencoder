# face-recognition-autoencoder  

face-recognition is one the primary and yet simple algorithm to tackle when starting Deep Learning. In this face-recognition repository, we will see face-recognition which uses only autoencoder and it's types. Our dataset consist of eight labels means we have eight faces to recognize. Each label have 32 images only. If need we can increase the size of dataset.

# unsupervised pretraining  
As we know, if we have not limited labelled data but have much unlabelled data then it is good to use a unsupervised pretraining technique for training. We use this unsupervised pretraining using autoencoder. Autoencoder are of different types Stacked, Denoising, Sparse, Variational, others. Any of autoencoder maybe, it motive is simple to learn the efficient pattern in the training set. There is "coding layer", which does this work, it learns the efficient data representation. Later, we use this autoencoder or just use lower layers of autoencoder. 

unsupervised_pretraining_using_autoencoder_and_machine_learning.py - this consist of unsupervised pretraining as we discussed above. Training unsupervised autoencoder with unlabelled data and then output of autoencoder is trained as input with labelled data for any machine-learning algorithm like SVC, SGDClassifier, other classification algorithms. 

unsupervised_pretraining_using_autoencoder_and_fine_tuning.py - Training unsupervised autoencoder is similar to above one. But we then use the lower layers of autoencoder, fixed the weights, then we fine-tune this existing model, training it with labelled data.

training_one_autoencoder_at_a_time.py - This is just one way to train an autoencoder. It's just a sample algorithm, you can study deep about it, if you want.
