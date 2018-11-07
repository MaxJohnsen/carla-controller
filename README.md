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
 The Carla Controller is used to control the simulator and can record the driving data to disk. This data can be used to train the model later on.

### Start Carla Controller 

1. Navigate to the folder you donwloaded and extracted _Carla v. 0.8.2_ from
2. Run the server:
    ```
    CarlaUE4.exe -carla-server -windowed
    ```
3. The server is now waiting for a client to connect 
4. In a new window navigate to the repo _e2e-self-driving-car_
5. Run the carla controller: 
    ```
    python carla_controller.py
    ```

    For help run: 
    ```
    python carla_controller.py --help
    ```  
6. Congratulations, you are running the Carla Controller. Read the _Actions_-section to learn how to control it  

### Arguments 

You can run the Carla Controller with different arguments to customize the controller: 

Argument | Description | Opotions 
--- | --- | ---
-v, --verbos e| Prints debug info | 
--host | IP of the host server | default is localhost
-p, --port | TCP port to listen to | default is 2000
-o, --output | Name of directory to save driving data to

Example: `python carla_controller.py -v -p 2001 -o "test1"` will listen at port 2001, print debug information and save all driving data to a folder called _test1_

 ### Actions  

 When running the Carla Controller following actions can be taken: 

Key | Action
--- | ---
 A, ← | Steer left 
 D, → | Steer right 
 W, ↑ | Throttle 
 S, ↓ | Break
 Space | Handbreak
 Q | Toggle reverse 
 P | Toggle autopilot
 R | Toggle driving data recording
 E | Start new episode 

### Data logging 

To be able to log the driving data you need to run the carla controller with an output argument: `python carla_controller.py -o "output-folder"`. 

Every time you press the R-key while driving in the simulator, the driving data will now be saved in a folder named _output-folder_. The data will be saved as following: 

Each episode will get a folder with a timestamp as the name. The episode folder will have one folder with images and one drivng log. 

The structure looks like this: 

- output-folder 
    - YYYY-MM-DD HH:mm:SS
        - images 
        - driving_log.csv
    - YYYY-MM-DD HH:mm:SS
        - images 
        - driving_log.csv
