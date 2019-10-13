#Policy gradients and actor critic methods

To run the training, run the command
```bash
bash run.sh
```
run.sh lists all the parameters. The parameters to pass in are --lr (learning rate), --gamma (discount factor), and --exp (a string to mark the experiment)

All plots will be saved at a location ./results. Make this folder in /policy_gradient_learning

### Dependencies
Swig == 3.0.8 (need to build from source on ubuntu 14.04)
