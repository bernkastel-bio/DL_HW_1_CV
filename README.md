# dl_hw1_cv
Trying to achieve >90% accuracy on galaxy10 dataset...

Model: [Wavemix](https://github.com/pranavphoenix/WaveMix/tree/main)

The very first try was yolov8 classifier, but I didn't saved the result( \
The first nd second tries with WaveMix is in the notebooks, but I the best accuracy that I got on test was about 83% and I spent 3,33 hours of training. I have a system of interactive sessions on server for juphub and for gpu there is a strict limit for 4 hours, so I have made a python scripts for the following experiments to run them using slurm partirion on cluster.

I also uploaded here weights for trained models, accuracy on test set is signed in .pth file names.
The final script denoted in its filename as try3.

## Requirements

I had been using conda env with packages signed in requirements.txt
