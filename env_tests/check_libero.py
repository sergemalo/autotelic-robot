# This script was taken from the LIBERO ReadMe file
# I only added some print statements to follow the execution
import os

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path


print("Getting LIBERO benchmark dict...")
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_spatial" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()


task_id = 0
print(f"Retrieve Task {task_id} of task_suite {task_suite_name}:")
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"Task name        : {task_name}")
print(f"Task description : {task_description}")
print(f"Task BDDL file   : {task_bddl_file}")

print(f"Creating LIBERO environment...")
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()


print(f"Init LIBERO state...")
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

print(f"Run 10 LIBERO env steps...")
dummy_action = [0.] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
env.close()

print(f"DONE !")
