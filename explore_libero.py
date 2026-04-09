# This script was taken from the LIBERO ReadMe file
# I only added some print statements to follow the execution
import argparse
from datetime import datetime
import os
import logging
import imageio
import numpy as np

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path


logger      = logging.getLogger(__name__)
date_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir  = f"output/{date_prefix}_explore_libero"
log_file    = f"{output_dir}/run.log"
os.mkdir(output_dir)


def init_logging(log_level: str):
    """
    Initialize logging with both console and file handlers.
     - Console logs are filtered by the specified log level.
     - File logs capture everything at DEBUG level for detailed analysis.
     - Log file is saved in the output directory with a timestamped name.
    """
    # Tricky: using root logger to ensure all logs (including from imported modules) are captured and directed to our handlers.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Global minimum level

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())

    # --- File handler ---
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Attach handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)    

def main():
    parser = argparse.ArgumentParser(description="LIBEROE Explore Script")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    args = parser.parse_args()
    init_logging(args.log_level)

    logger.info("Starting LIBERO exploration script...")

    logger.info("Getting LIBERO benchmark dict...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_spatial" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()


    task_id = 0
    logger.info(f"Retrieve Task {task_id} of task_suite {task_suite_name}:")
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    logger.info(f"Task name        : {task_name}")
    logger.info(f"Task description : {task_description}")
    logger.info(f"Task BDDL file   : {task_bddl_file}")

    logger.info(f"Creating LIBERO environment...")
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 224,
        "camera_widths": 224
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()





    logger.info(f"Init LIBERO state...")
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])

    frames = [obs["agentview_image"][::-1, :, :]]
    logger.info(f"Run 100 LIBERO env steps...")
    #dummy_action = [0.2, 0, 0, 0, 0, 0, 0] # Towards camera
    #dummy_action = [0.0, 0.2, 0, 0, 0, 0, 0] # Towards right
    #dummy_action = [0.0, 0.0, 0.2, 0, 0, 0, 0] # Towards Up
    #dummy_action = [0.0, 0.0, 0.0, 0.2, 0, 0, 0] # Rotate counter-clockwise from the view of the agent
    #dummy_action = [0.0, 0.0, 0.0, 0.0, 0.2, 0, 0] # Rotate clockwise from the view from the left of the scene
    #dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0] # Rotate counter-clockwise from the view from the top of the scene
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2] # Positive: close gripper, Negative: open gripper
    for step in range(100):

        obs, reward, done, info = env.step(dummy_action)
        frames.append(obs["agentview_image"][::-1, :, :])
        #dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

    env.close()

    video_path_ = os.path.join(output_dir, "libero_exploration.mp4")
    if len(frames) > 0:
        imageio.mimsave(video_path_, frames, fps=20)

    logger.info(f"DONE !")


if __name__ == "__main__":
    main()