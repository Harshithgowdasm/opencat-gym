# OpenCat Gym
A reinforcement learning environment for gait generation and step-climbing tasks on the Petoi Bittle quadruped robot. Built using Stable-Baselines3 and PyBullet, OpenCat Gym provides an efficient framework for training and deploying RL policies for autonomous locomotion.

## Installation and Usage

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the repository:**
   ``` bash
   git clone https://github.com/SuchethShenoy/opencat-gym.git
   cd opencat-gym
   ```
2. **Create a virtual environment:** 
    ``` bash
    python3 -m venv venv
    source venv/bin/activate  
    ```
3. **Install the required packages::** 
    ``` python
    pip install -r requirements.txt 
    ```
4. **Start with training:** 
    The training process automatically saves the best models. To train for step climbing, change the Gym environment to o`opencat_step_gym_env.py` and use any RL algorithm supported by Stable-Baselines3.

    ``` python
    python train_with_callback.py
    ```
4. **Visulize the results:** 
    After training, visualize the results using the saved policy models:
    ``` python
    python visualize_policy_directory.py
    ```

### Results
<img src=animations/best_model_gait_TD3_.gif width="400" />  <img src=animations/best_model_single_step_PPO.gif width="400" />
Gait Generation (TD3)                                           Single Step Climbing (PPO)


<img src=animations/best_model_double_step_PPO.gif width="400" /> <img src=animations/best_model_triple_step_PPO.gif width="400" />
Double Step Climbing (PPO)                          Triple Step Climbing (PPO)



## Links
For more information on the reinforcement training implementation: https://stable-baselines3.readthedocs.io/en/master/index.html \
And for the simulation environment please refer to: https://pybullet.org/wordpress/ \
The API for creating the training environment: https://gymnasium.farama.org/

## Related Work
The reward and penalty functions are based on: https://www.nature.com/articles/s41598-023-38259-7 \
Including a joint angle history was inspired by: https://www.science.org/doi/10.1126/scirobotics.aau5872
