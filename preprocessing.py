import numpy as np
import os
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Data generator for loading seismic spectrograms and response spectra
class SeismicDataGenerator(keras.utils.Sequence):
    def __init__(self, hdf5_path, images_folder, batch_size=16, image_type=None, 
                 indices=None, shuffle=True, image_size=(128, 512), trim_time="3s"):
        """
        Data generator for loading seismic spectrograms and response spectra
        
        Args:
            hdf5_path: Path to the HDF5 file
            images_folder: Base folder containing spectrograms
            batch_size: Batch size for training
            image_type: Optional folder name for spectrograms (if None, will be constructed from trim_time)
            indices: List of indices to use (for train/val/test split)
            shuffle: Whether to shuffle data between epochs
            image_size: Target size for the images (height, width)
            trim_time: Trim time to use (e.g., "2s", "3s", "25s") if image_type not provided
        """
        self.hdf5_path = hdf5_path
        self.images_folder = images_folder
        self.batch_size = batch_size
        
        # If image_type is not explicitly provided, construct it from trim_time
        if image_type is None:
            self.image_type = f"trimmed_{trim_time}"
        else:
            self.image_type = image_type
            
        self.image_size = image_size
        self.shuffle = shuffle
        
        print(f"Using image type: {self.image_type}")
        
        # Open HDF5 file and get metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            self.osc_periods = f['osc_periods'][:]
            metadata = f['image_metadata'][:]
            self.event_ids = [event_id.decode('utf-8') for event_id in metadata['event_id']]
            
            # Print the first few event IDs for debugging
            print(f"First few event IDs: {self.event_ids[:5]}")
            
            # Construct paths directly from event_ids with the correct format
            self.image_paths = []
            for event_id in self.event_ids:
                # Use the event_id directly for constructing the path
                img_path = os.path.join(self.images_folder, self.image_type, f"{event_id}.png")
                self.image_paths.append(img_path)
            
            # Print a few paths for debugging
            print(f"First few image paths: {self.image_paths[:3]}")
            
            # Check if these paths exist
            for path in self.image_paths[:5]:
                print(f"Path {path} exists: {os.path.exists(path)}")
        
        # Use provided indices or all indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.event_ids)))
            
        self.num_samples = len(self.indices)
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Get batch at index"""
        # Generate indices for this batch
        batch_indices = self.indices_list[index * self.batch_size:
                                        min((index + 1) * self.batch_size, self.num_samples)]
        
        # Allocate batch arrays
        batch_images = np.zeros((len(batch_indices), self.image_size[0], self.image_size[1], 3), 
                            dtype=np.float32)
        batch_spectra = np.zeros((len(batch_indices), len(self.osc_periods)), dtype=np.float32)
    
        # Load data for this batch
        for i, idx in enumerate(batch_indices):
            # Get original index and event_id
            orig_idx = self.indices[idx]
            event_id = self.event_ids[orig_idx]
            
            # Get image path
            img_path = self.image_paths[orig_idx]
            
            # Try multiple path formats if the file doesn't exist
            if not os.path.exists(img_path):
                possible_paths = [
                    # Try removing .png if it was added
                    img_path[:-4] if img_path.endswith('.png') else img_path,
                    # Try with .png if it wasn't added
                    img_path + '.png' if not img_path.endswith('.png') else img_path,
                    # Try event_XXXX format (with 4 digits)
                    os.path.join(self.images_folder, self.image_type, f"{event_id}.png"),
                    # Try event_X format (without padding zeros)
                    os.path.join(self.images_folder, self.image_type, f"event_{event_id.split('_')[1].lstrip('0')}.png"),
                ]
                
                # Check each possible path
                found_path = False
                for path in possible_paths:
                    if os.path.exists(path):
                        img_path = path
                        found_path = True
                        break
                
                if not found_path:
                    print(f"WARNING: Cannot find image for event {event_id}. Tried paths:")
                    for path in possible_paths:
                        print(f"  - {path} (exists: {os.path.exists(path)})")
            
            try:
                # Open the image
                img = Image.open(img_path)
                # Convert RGBA to RGB if needed
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                img = img.resize(self.image_size[::-1])  # PIL uses (width, height)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                # Ensure we have 3 channels
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]  # Take only RGB channels
                
                batch_images[i] = img_array
                
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                # Use a blank image as fallback
                batch_images[i] = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)
            
            # Load response spectra
            try:
                with h5py.File(self.hdf5_path, 'r') as f:
                    batch_spectra[i] = f[f"{event_id}/response_spectra/spec_accel"][:]
            except Exception as e:
                print(f"Error loading spectra for {event_id}: {str(e)}")
                # Use zeros as fallback
                batch_spectra[i] = np.zeros(len(self.osc_periods), dtype=np.float32)
        
        # Normalize spectra (important for VAE training)
        batch_spectra = self._normalize_spectra(batch_spectra, is_log=True)
        
        return {"input_image": batch_images}, {"output_spectra": batch_spectra, 
                                            "z_mean": np.zeros((len(batch_indices), self.latent_dim)),
                                            "z_log_var": np.zeros((len(batch_indices), self.latent_dim))}

    def getitem(self, index):
        """Implements the abstract method from the parent class"""
        return self.__getitem__(index)
    
    def on_epoch_end(self):
        """Updates indices after each epoch"""
        self.indices_list = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices_list)
    
    def _normalize_spectra(self, spectra, is_log=True):
        """
        Normalize response spectra for better training
        
        Args:
            spectra: Input spectra data
            is_log: Whether the data is already in logarithmic form
        """
        if is_log:
            # Data is already logarithmic, so skip log transformation
            # Apply min-max scaling directly
            spectra_min = np.min(spectra, axis=1, keepdims=True)
            spectra_max = np.max(spectra, axis=1, keepdims=True)
            
            # Create range mask correctly for 2D array
            # The mask should have the same shape as spectra
            range_diff = spectra_max - spectra_min
            # Create mask that has the same shape as spectra
            range_mask = np.broadcast_to(range_diff > 1e-8, spectra.shape)
            
            # Initialize normalized array
            spectra_normalized = np.zeros_like(spectra)
            
            # Apply normalization using proper broadcasting
            # Handle division only where range is sufficient
            safe_range = range_diff.copy()
            safe_range[safe_range <= 1e-8] = 1.0  # Avoid division by zero
            
            # Normalize
            spectra_normalized = (spectra - spectra_min) / safe_range
            
            # For cases where range is too small, just use the original values
            spectra_normalized[~range_mask] = spectra[~range_mask]
            
            return spectra_normalized
        else:
            # Original normalization for non-logarithmic data
            spectra = np.log1p(spectra)
            spectra_min = np.min(spectra, axis=1, keepdims=True)
            spectra_max = np.max(spectra, axis=1, keepdims=True)
            spectra = (spectra - spectra_min) / (spectra_max - spectra_min + 1e-8)
            return spectra
    
    @property
    def latent_dim(self):
        """Latent dimension for the VAE"""
        return 256  # Adjust as needed

# Utility functions for dataset validation
def check_dataset_integrity(hdf5_path, images_folder, trim_time="3s"):
    """
    Check if all files referenced in the HDF5 exist and are accessible
    
    Args:
        hdf5_path: Path to the HDF5 file
        images_folder: Base folder containing spectrograms
        trim_time: Trim time to use (e.g., "2s", "3s", "25s")
        
    Returns:
        tuple: (found_files, missing_files) lists of event IDs
    """
    # Use specified trim time for the path
    image_type = f"trimmed_{trim_time}"
    
    print(f"\n===== DATASET INTEGRITY CHECK ({image_type}) =====")
    with h5py.File(hdf5_path, 'r') as f:
        metadata = f['image_metadata'][:]
        event_ids = [event_id.decode('utf-8') for event_id in metadata['event_id']]
        
        missing_files = []
        found_files = []
        for i, event_id in enumerate(event_ids):
            # Try the path structure with the specified trim time
            img_path = os.path.join(images_folder, image_type, f"{event_id}.png")
            
            if os.path.exists(img_path):
                found_files.append(event_id)
            else:
                missing_files.append(event_id)
            
            # Print progress every 1000 files
            if (i+1) % 1000 == 0 or i == len(event_ids) - 1:
                print(f"Checked {i+1}/{len(event_ids)} files...")
        
        if missing_files:
            print(f"\nWARNING: {len(missing_files)} out of {len(event_ids)} files are missing")
            print(f"First few missing files: {missing_files[:5]}")
        else:
            print(f"\nAll {len(event_ids)} files exist!")
        
        print("===== END OF INTEGRITY CHECK =====\n")
        
        return found_files, missing_files

# def validate_log_data(hdf5_path):
#     """Validate that logarithmic data doesn't contain invalid values"""
#     print("Validating logarithmic response spectra data...")
    
#     with h5py.File(hdf5_path, 'r') as f:
#         # Get all event IDs (excluding special datasets)
#         event_ids = [key for key in f.keys() if key not in ['osc_periods', 'image_metadata']]
        
#         invalid_count = 0
#         total_count = len(event_ids)
        
#         for i, event_id in enumerate(event_ids):
#             # First check if the path exists before trying to access it
#             if 'response_spectra' in f[event_id] and 'spec_accel' in f[event_id]['response_spectra']:
#                 spectra = f[f"{event_id}/response_spectra/spec_accel"][:]
                
#                 # Check for NaN, inf, or negative values (invalid for log data)
#                 if np.any(np.isnan(spectra)) or np.any(np.isinf(spectra)):
#                     print(f"Warning: NaN or Inf values found in {event_id}")
#                     invalid_count += 1
#             else:
#                 print(f"Warning: Response spectra data not found for {event_id}")
#                 invalid_count += 1
            
#             # Print progress every 1000 files or at the end
#             if (i+1) % 1000 == 0 or i == total_count - 1:
#                 print(f"Validated {i+1}/{total_count} events...")
        
#         if invalid_count > 0:
#             print(f"\nWARNING: {invalid_count} out of {total_count} events have invalid data")
#             valid = False
#         else:
#             print(f"\nAll {total_count} events have valid data!")
#             valid = True
            
#         return valid