# autoencoder_latent_space_visualization

This repo contains training of custom autoencoder on mnist dataset(28x28x1) and visualization of its latent space (d-128).

## The Autoencoder model architecture is shown below:
<p align="center">
  <img src="https://github.com/tshr-d-dragon/autoencoder_latent_space_visualization/blob/main/images/autoencoder_model.png?raw=true" alt="Autoencoder"/>
</p>

## Prediction on sample (10) Test dataset: 
#### First row is the ground-truths and second row is predictions
![Image1](https://github.com/tshr-d-dragon/autoencoder_latent_space_visualization/blob/main/images/prediction.png)

## Visualization of mnist dataset vector with d-784 (28x28x1) to d-2 (using TSNE):
![Image2](https://github.com/tshr-d-dragon/autoencoder_latent_space_visualization/blob/main/images/original/img_784_tsne_perplexity_50.png)

## Visualization of latent space vector of mnist dataset with d-128 to d-2 (using TSNE):
![Image3](https://github.com/tshr-d-dragon/autoencoder_latent_space_visualization/blob/main/images/latent_space_vector/img_128_tsne_perplexity_50.png)
