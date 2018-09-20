# Simultaneous Separation and Recognition of sequences of digits

## Flow
- Cochlea (ams1c) 2 channels: SpikeSeparation
- Reconstruction model: from spikes to mfccs
- Recognition model: from mff to words (trained with noise in features)

## Steps
1. Record new dataset (TIDIGITS) with 2 ears and random delays [DONE]
2. Run spike separation alg [DONE]
3. Find model for reconstruction [TODO]
4. Train sequence model with noise in input features [TODO]
5. Test [TODO]
 
## Details on Steps
### Step 1 & 2
Done recording 6000 samples for training and 2000 for testing (different speakers).
Delays look pretty good and ITD and assignment as well so emulating the delays works fine 
(for future references). In order to increase the number of assigned spikes I had to go through
an ITD recalibration step (which could be an entire paper on its own) where I estimate ho much 
the delays on single channels are wrong and shift them. this allows me to gain many more 
spikes which will be needed for reconstruction (next step).

### Step 3
The main problem here is the kind of model and what to train on. One possibility is an end-to-end 
retraining of an already trained TIDIGITS model that also considers reconstruction (because why not)
By using the gt I can impose the correct reconstruction. 
But the main idea was noise adaptation: I reconstruct the fbank as best as I can and then I retrain
the recon model to be robust to that kind of noise.

    Experiments step 3:
    - MODE 1: Directly using separated spikes 
    - MODE 2: using projected spk2log model
    - MODE 3: Do noise degradation during training (same distribution)
    - MODE 4: End-to-end reconstruction and recognition