# Simultaneous Separation and Recognition of sequences of digits

## Flow
- Cochlea (ams1c) 2 channels: SpikeSeparation
- Reconstruction model: from spikes to mfccs
- Recognition model: from mff to words (trained with noise in features)

## Steps
- Record new dataset (TIDIGITS) with 2 ears and random delays
- Run spike separation alg
- Find model for reconstruction 
- Train sequence model with noise in input features
- Test
 
