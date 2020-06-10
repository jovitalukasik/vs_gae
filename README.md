# A Variational-Sequential Graph Autoencoder for Neural Architecture Performance Prediction
===============================================================================

Abstract
-----

In computer vision research, the process of automating architecture engineering, Neural Architecture Search (NAS), has gained substantial interest. In the past, NAS was hardly accessible to researchers without access to large-scale compute systems, due to very long compute times for the recurrent search and evaluation of new candidate architectures. The NAS-Bench-101 dataset facilitates a paradigm change towards classical methods such as supervised learning to evaluate neural architectures. In this paper, we propose a graph encoder built upon Graph Neural Networks (GNN). We demonstrate the effectiveness of the proposed encoder on NAS performance prediction for seen architecture types as well an unseen ones (i.e., zero shot prediction). We also provide a new variational-sequential graph autoencoder (VS-GAE) based on the proposed graph encoder. The VS-GAE is specialized on encoding and decoding graphs of varying length utilizing GNNs. Experiments on different sampling methods show that the embedding space learned by our VS-GAE increases the stability on the accuracy prediction task. 


Example
------------
The NAS-Bench-101 dataset is split into 70%/20%/10% training-,test- and validation set and can be found in enc_data.zip.
To train the encoder and the performance prediction for the whole trainings dataset for 100 epochs run

  `python enc_perf_prediction.py`
  
To train the encoder and the performance prediction for 10% equidistant sampled data run

   `python enc_perf_prediction.py --training_size 10 --sampling 'even'`
   
The data for the extrapolation task (extrapolate to 6/extrapolate to 7) can be found in extra_6.zip/extra_7.zip. 
    
To train the variational autoencoder for 90% of the dataset (vae_data.zip) for 300 epochs and test the autoencoder abilities run

   `python train_gae.py --test`

 Store all unzip datasets in the folder "data/". 

Reference
---------
For more information, see the paper


D.Friede, J.Lukasik, H.Stuckenschmidt, M.Keuper. A Variational-Sequential Graph Autoencoder for Neural Architecture Performance Prediction. *arXiv preprint arXiv: 1912.05317*, 2019
