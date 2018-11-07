# End-to-End Self-Driving Car

## Setup Guide (Windows)

1. Download and install Anaconda
2. Download and extract [Carla v. 0.8.2](https://github.com/carla-simulator/carla/releases/tag/0.8.2)
3. Clone and navigate to the repo:

    ```
    git clone https://github.com/MaxJohnsen/e2e-self-driving-car && cd e2e-self-driving-car
    ```

4. Install dependencies in a new Conda environment:

    ```
    conda env create -f environment.yml
    ```

5. Activate new environment:

    ```
    conda activate e2e-sdc
    ```
 
 ## Carla Controller 


For help run: 
```
python carla_controller.py --help
```  

 ### Functionality 

 When running the Carla Controller following actions can be taken: 

Key | Action
--- | ---
 A or ← | Steer left 
 D or → | Steer right 
 W or ↑ | Throttle 
 S or ↓ | Break
 Q | Toggle reverse 
 Space | Handbreak
 R | Start new episode 


