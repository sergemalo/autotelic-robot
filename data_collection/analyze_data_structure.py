"""
LIBERO Dataset Explorer
This script helps you:
1. Load a LIBERO dataset
2. Sample and visualize random trajectories
3. Understand the dataset structure and dimensions
"""

import h5py
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import os

def explore_dataset_structure(hdf5_path):
    """
    Explore and print the structure of a LIBERO dataset
    """
    print(f"\n{'='*80}")
    print(f"Exploring dataset: {hdf5_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(hdf5_path, 'r') as f:
        print("Top-level keys in dataset:")
        print(f"  {list(f.keys())}\n")
        
        # Explore data group
        if 'data' in f:
            data = f['data']
            demos = list(data.keys())
            num_demos = len(demos)
            print(f"Number of demonstrations: {num_demos}")
            print(f"Demo keys: {demos[:5]}... (showing first 5)\n")
            
            # Pick first demo to explore structure
            demo_0 = data[demos[0]]
            print(f"Structure of {demos[0]}:")
            print(f"  Keys: {list(demo_0.keys())}\n")
            
            # Explore each key in detail
            for key in demo_0.keys():
                item = demo_0[key]
                if isinstance(item, h5py.Dataset):
                    print(f"  {key}:")
                    print(f"    Shape: {item.shape}")
                    print(f"    Dtype: {item.dtype}")
                    if len(item.shape) > 0 and item.shape[0] > 0:
                        print(f"    Sample value: {item[0]}")
                    print()
                elif isinstance(item, h5py.Group):
                    print(f"  {key}: (Group)")
                    for subkey in item.keys():
                        subitem = item[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            print(f"    {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
                    print()
        
        # Explore mask if it exists
        if 'mask' in f:
            mask = f['mask']
            print(f"Mask information:")
            print(f"  Keys: {list(mask.keys())}")
            for key in mask.keys():
                print(f"  {key}: {mask[key][()]}")
            print()

def get_random_trajectory(hdf5_path):
    """
    Sample a random trajectory from the dataset
    Returns the demo name and the trajectory data
    """
    with h5py.File(hdf5_path, 'r') as f:
        demos = list(f['data'].keys())
        random_demo = random.choice(demos)
        print(f"\nSelected random trajectory: {random_demo}")
        return random_demo

def visualize_trajectory(hdf5_path, demo_name, output_dir="output/trajectory_samples", save_video=True):
    """
    Visualize a trajectory by extracting frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        demo = f['data'][demo_name]
        
        # Get trajectory length
        traj_length = demo['actions'].shape[0]
        print(f"Trajectory length: {traj_length} steps")
        
        # Check available observation types
        obs_keys = list(demo['obs'].keys())
        print(f"Available observations: {obs_keys}")
        
        # Try to find image observations
        image_keys = [k for k in obs_keys if 'image' in k.lower() or 'agentview' in k.lower() or 'robot0' in k.lower()]
        
        if not image_keys:
            print("No image observations found!")
            return None
        
        print(f"Using image key: {image_keys[0]}")
        images = demo['obs'][image_keys[0]][:]
        
        print(f"Image shape: {images.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Image range: [{images.min()}, {images.max()}]")
        
        # Extract frames
        frames = []
        for i in range(min(traj_length, len(images))):
            img = images[i]
            
            # Handle different image formats
            if img.dtype == np.uint8:
                img_pil = Image.fromarray(img)
            else:
                # Normalize to 0-255 if needed
                img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_normalized)
            
           
            frames.append(np.array(img_pil))
            
        
        # Create a video using imageio
        if save_video:
            try:
                import imageio
                video_path = os.path.join(output_dir, f"trajectory_{demo_name}.mp4")
                imageio.mimsave(video_path, frames, fps=10)
                print(f"Saved video to {video_path}")
            except ImportError:
                print("imageio not available for video creation. Install with: pip install imageio[ffmpeg]")
        

        
        return frames

def analyze_trajectory_data(hdf5_path, demo_name):
    """
    Analyze the data dimensions and types in a trajectory
    """
    print(f"\n{'='*80}")
    print(f"Analyzing trajectory: {demo_name}")
    print(f"{'='*80}\n")
    
    with h5py.File(hdf5_path, 'r') as f:
        demo = f['data'][demo_name]
        
        # Actions
        actions = demo['actions'][:]
        print("Actions:")
        print(f"  Shape: {actions.shape}")
        print(f"  Dtype: {actions.dtype}")
        print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"  First action: {actions[0]}")
        print()
        
        # States (if available)
        if 'states' in demo:
            states = demo['states'][:]
            print("States:")
            print(f"  Shape: {states.shape}")
            print(f"  Dtype: {states.dtype}")
            print(f"  Range: [{states.min():.3f}, {states.max():.3f}]")
            print()
        
        # Rewards
        if 'rewards' in demo:
            rewards = demo['rewards'][:]
            print("Rewards:")
            print(f"  Shape: {rewards.shape}")
            print(f"  Sum: {rewards.sum()}")
            print(f"  Non-zero steps: {np.count_nonzero(rewards)}")
            print()
        
        # Dones
        if 'dones' in demo:
            dones = demo['dones'][:]
            print("Dones:")
            print(f"  Shape: {dones.shape}")
            print(f"  Terminal steps: {np.where(dones)[0]}")
            print()
        
        # Observations
        print("Observations:")
        obs = demo['obs']
        for key in obs.keys():
            obs_data = obs[key][:]
            print(f"  {key}:")
            print(f"    Shape: {obs_data.shape}")
            print(f"    Dtype: {obs_data.dtype}")
            if 'image' not in key.lower():
                print(f"    Range: [{obs_data.min():.3f}, {obs_data.max():.3f}]")
                if obs_data.shape[0] > 0:
                    print(f"    First value: {obs_data[0][:10] if len(obs_data[0]) > 10 else obs_data[0]}")
            print()

def main():
    # Set the path to your LIBERO dataset
    # Adjust this path based on where you downloaded the dataset
    dataset_path = "/content/autotelic-robot/src/libero/libero/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
    
    # You can also manually specify a path
    # dataset_path = "/path/to/your/libero_spatial_demo.hdf5"
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found at {dataset_path}")
       
        return
    
    # 1. Explore dataset structure
    explore_dataset_structure(dataset_path) # This will print the structure and dimensions of the first demo in the dataset
    
    # 2. Get a random trajectory
    demo_name = get_random_trajectory(dataset_path)
    
    # 3. Analyze trajectory data
    analyze_trajectory_data(dataset_path, demo_name) 
    
    # 4. Visualize the trajectory
    print("\nVisualizing trajectory...")
    frames = visualize_trajectory(dataset_path, demo_name)
    
    print("\n" + "="*80)
    print("Exploration complete!")
    print("="*80)

if __name__ == "__main__":
    main()