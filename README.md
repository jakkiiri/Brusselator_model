This branch explores the method of the Physics Informed Neural Network in solving the Brusselator Model.

Detailed documentation on architecture and training method are in Documentation File.
The old_scripts folder include loose scripts used for sanity testing and experimental exploration of the PINN on local computer. 

If you want to attempt the sanity test locally run pinn_training_multi.py from the old_scripts folder.

1. Create a Venv: python3 -m venv venv
2. Activate Venv: ./venv/Scripts/Activate
3. Cd into old_scripts: cd old_scripts
4. Install requirements: pip install -r requirements.txt
5. Run the Python File: python run pinn_training.py or python run pinn_training_multi.py

If you want to attempt training with hpc you can run the slurm file in hpc_scripts along with your VM setup properly with the correct dependecy.
The hpc_scripts folder include the final version of the training script used for training the PINN model on a remote High Performing Computing Node provided by Canada Computing Alliance.

The config file contains the most updated configuration used to train the final iteration of the model.  