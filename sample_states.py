"""
LIBERO State Sampler
Sample starting and goal states from LIBERO trajectories
"""

import h5py
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import os

class LIBEROStateSampler:
    """
    A class to sample starting and goal states from LIBERO datasets
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.demos = None
        self._load_demo_list()
    
    def _load_demo_list(self):
        """Load the list of available demonstrations"""
        with h5py.File(self.dataset_path, 'r') as f:
            self.demos = list(f['data'].keys())
        print(f"Loaded {len(self.demos)} demonstrations from dataset")
    
    def sample_trajectory(self):
        """Sample a random trajectory from the dataset"""
        demo_name = random.choice(self.demos)
        return demo_name
    
    def get_trajectory_length(self, demo_name):
        """Get the length of a specific trajectory"""
        with h5py.File(self.dataset_path, 'r') as f:
            traj_length = f['data'][demo_name]['actions'].shape[0]
        return traj_length
    
    def sample_start_goal_states(self, demo_name=None, min_distance=5, goal_after_start=True):
        """
        Sample starting and goal states from a trajectory
        
        Args:
            demo_name: Name of the demo. If None, samples randomly
            min_distance: Minimum number of steps between start and goal
            goal_after_start: If True, goal must come after start. If False, any state can be goal
        
        Returns:
            dict with start_idx, goal_idx, demo_name, and state data
        """
        if demo_name is None:
            demo_name = self.sample_trajectory()
        
        traj_length = self.get_trajectory_length(demo_name)
        
        # Sample start state and goal state indices
        if goal_after_start:
            # Start must leave room for goal
            max_start = traj_length - min_distance - 1
            start_idx = random.randint(0, max(0, max_start))
            # Goal must be at least min_distance steps after start
            goal_idx = random.randint(
                min(start_idx + min_distance, traj_length - 1),
                traj_length - 1
            )
        else:
            # Start and goal can be any states
            start_idx = random.randint(0, traj_length - 1)
            # Sample goal ensuring minimum distance
            possible_goals = list(range(0, max(0, start_idx - min_distance))) + \
                           list(range(min(start_idx + min_distance, traj_length), traj_length))
            if not possible_goals:
                # If trajectory too short, just use endpoints
                start_idx = 0
                goal_idx = traj_length - 1
            else:
                goal_idx = random.choice(possible_goals)
        
        # Load the actual state data
        with h5py.File(self.dataset_path, 'r') as f:
            demo = f['data'][demo_name]
            
            # Get state information
            start_state = {}
            goal_state = {}
            
            # Actions
            start_state['action'] = demo['actions'][start_idx]
            goal_state['action'] = demo['actions'][goal_idx] if goal_idx < demo['actions'].shape[0] else None
            
            # States (if available)
            if 'states' in demo:
                start_state['state'] = demo['states'][start_idx]
                goal_state['state'] = demo['states'][goal_idx]
            
            # Observations
            start_state['obs'] = {}
            goal_state['obs'] = {}
            for key in demo['obs'].keys():
                start_state['obs'][key] = demo['obs'][key][start_idx]
                goal_state['obs'][key] = demo['obs'][key][goal_idx]
        
        return {
            'demo_name': demo_name,
            'trajectory_length': traj_length,
            'start_idx': start_idx,
            'goal_idx': goal_idx,
            'start_state': start_state,
            'goal_state': goal_state,
            'distance': abs(goal_idx - start_idx)
        }
    
    def visualize_start_goal(self, sample_result, save_path=None):
        """
        Visualize the starting and goal states side by side
        
        Args:
            sample_result: Output from sample_start_goal_states()
            save_path: Optional path to save the visualization
        """
        # Find image observation key
        obs_keys = list(sample_result['start_state']['obs'].keys())
        image_keys = [k for k in obs_keys if 'image' in k.lower() or 'agentview' in k.lower() or 'robot0' in k.lower()]
        
        if not image_keys:
            print("No image observations available for visualization")
            return
        
        image_key = image_keys[0]
        start_img = sample_result['start_state']['obs'][image_key]
        goal_img = sample_result['goal_state']['obs'][image_key]
        
        # Convert to PIL images
        if start_img.dtype != np.uint8:
            start_img = ((start_img - start_img.min()) / (start_img.max() - start_img.min()) * 255).astype(np.uint8)
            goal_img = ((goal_img - goal_img.min()) / (goal_img.max() - goal_img.min()) * 255).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(start_img)
        axes[0].set_title(f"Start State (step {sample_result['start_idx']})", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(goal_img)
        axes[1].set_title(f"Goal State (step {sample_result['goal_idx']})", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.suptitle(
            f"Demo: {sample_result['demo_name']} | "
            f"Distance: {sample_result['distance']} steps | "
            f"Trajectory length: {sample_result['trajectory_length']}",
            fontsize=12
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        plt.close()
    
    def print_sample_info(self, sample_result):
        """Print detailed information about a sampled start-goal pair"""
        print(f"\n{'='*80}")
        print(f"Sampled Start-Goal Pair")
        print(f"{'='*80}")
        print(f"Demo: {sample_result['demo_name']}")
        print(f"Trajectory length: {sample_result['trajectory_length']} steps")
        print(f"Start index: {sample_result['start_idx']}")
        print(f"Goal index: {sample_result['goal_idx']}")
        print(f"Distance: {sample_result['distance']} steps")
        print()
        
        print("Start State:")
        if 'state' in sample_result['start_state']:
            print(f"  State shape: {sample_result['start_state']['state'].shape}")
            print(f"  State sample: {sample_result['start_state']['state'][:5]}")
        print(f"  Action: {sample_result['start_state']['action']}")
        print(f"  Observations: {list(sample_result['start_state']['obs'].keys())}")
        print()
        
        print("Goal State:")
        if 'state' in sample_result['goal_state']:
            print(f"  State shape: {sample_result['goal_state']['state'].shape}")
            print(f"  State sample: {sample_result['goal_state']['state'][:5]}")
        if sample_result['goal_state']['action'] is not None:
            print(f"  Action: {sample_result['goal_state']['action']}")
        print(f"  Observations: {list(sample_result['goal_state']['obs'].keys())}")
        print(f"{'='*80}\n")

def main():
    # Set the path to your LIBERO dataset
    dataset_path = "/content/autotelic-robot/src/libero/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please update the dataset_path variable.")
        return
    
    # Create sampler and load dataset
    sampler = LIBEROStateSampler(dataset_path)

    os.makedirs("output/state_samples_visualizations", exist_ok=True)
    
    # Example 1: Sample with goal always after start
    print("\n" + "="*80)
    print("Example 1: Goal always after start (min_distance=5)")
    print("="*80)
    sample1 = sampler.sample_start_goal_states(min_distance=5, goal_after_start=True)
    sampler.print_sample_info(sample1)
    sampler.visualize_start_goal(sample1, save_path="output/state_samples_visualizations/start_goal_example1.png")
    
    # Example 2: Goal can be any state
    print("\n" + "="*80)
    print("Example 2: Goal can be before or after start (min_distance=10)")
    print("="*80)
    sample2 = sampler.sample_start_goal_states(min_distance=10, goal_after_start=False)
    sampler.print_sample_info(sample2)
    sampler.visualize_start_goal(sample2, save_path="output/state_samples_visualizations/start_goal_example2.png")
    

if __name__ == "__main__":
    main()