# LM-Bot-Battlebots
An LM solution to the common CompSci Battlebots challange, built using a mixture of Java and Python

To get started, setup a virtual environment (reccomended to be named `venv`):

```$ python3 -m venv venv```

Next, install requirements:

```$ ./venv/bin/pip3 install -r requirements.txt```

And finally the model training script:

```$ ./venv/bin/python3 botv2.py```

Training can only be run on Linux systems with CUDA drivers and a compatible CUDA GPU.
If required, this can be bypassed by adding the `device='cpu'` flag in the botv2.py 

This is tested and confirmed functional in Python 3.10, JVM 16.

To run hyperparameter data-gathering, execute `tune_yperparams.sh` which will run through the following model parameters, and log standard tensorflow training data dumps in `./logs_hyper_tuning/`:

```
max_grad_norm
vf_coef
ent_coef
clip_range
gae_lambda
gamma
linear_schedule (sensitivity/falloff)
```

The training script will iterate over those variables, results can be viewed in tensorboard:
```

$ source ./venv/bin/activate

$ tensorboard --logdir ./logs_hyper_tuning/ --bind_all

```

This agent uses Stable Baselines 3 implementation of the Proximal Policy Optimization algorithm. The core of the network is a MLP Policy, consisting of two seperate feed-forward neural networks:

Actor (Policy network): Decides which actions to take at any given time. Had three hidden layers with 256, 512, 256 neurons.

Critic (Value network): Provides state estimation. Has two hidden layers with 256, 256 neurons.

In total the agent model consists of 3690 neurons, and is capable of reaching a mean reward of 9-10 on a good training run while displaying immergent behaviour.