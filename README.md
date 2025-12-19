This branch explores the method of the Physics Informed Neural Network in solving the Brusselator Model.

Detailed documentation on model architecture and training methods of the final iteration are in the Documentation File.
Final training results, including loss history, training summary (epochs and loss data), and training/validation examples are in the training_results/pinn_ouputs folder
The old_scripts folder include loose scripts used for sanity testing and experimental exploration of the PINN on local computer. 
The findings from the old scripts folder only aided in the development and research of the final model and are not discussed explicitly in the report.

If you want to attempt the sanity test locally run pinn_training_multi.py/pinn_training.py from the old_scripts folder.

1. Create a Venv: python3 -m venv venv
2. Activate Venv: ./venv/Scripts/Activate
3. Cd into old_scripts: cd old_scripts
4. Install requirements: pip install -r requirements.txt
5. Run the Python File: python run pinn_training.py or python run pinn_training_multi.py

pinn_training.py:
-
- This python script is an initial exploration of the PINN architecture.
- The script trained a low scale model on a single sample of data with only physics and initial condition loss. This attempt at unsupervised learning proved to be difficult and supervised learning was introduced.
- Results showed good fit close to the initial conditions but immediate collapse after a bit of time.

pinn_training_multi.py:
-
- This python script is the second exploration of the PINN architecture that saw the implementation of supervised learning.
- The script trained a medium scale model on four samples of data with physics, initial condition, and data loss. This attempt acted as a sanity test that showed the PINN is able to converge and overfit on a limited sample of data.
- This also demonstrated that the solution space of the PINN is complex, where multiple loss descents were exhibited over a range of 20000 epochs.
- Ultimately this iteration informed the final iteration on the HPC to include supervised training, a large number of epochs (~100000), and the incorporation of Consine aneealing to navigate the complex solution space.

Scripts in the hpc_scripts were the scripts used in the final iteration of the model. 

If you want to attempt training with hpc you can run the slurm file in hpc_scripts along with your VM setup properly with the correct dependecies.
The hpc_scripts folder includes the final version of the training script used for training the PINN model on a remote High Performing Computing Node provided by Canada Computing Alliance.

The config.py file contains the most updated configuration used to train the final iteration of the model.  
