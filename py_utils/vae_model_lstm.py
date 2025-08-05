import os
import h5py
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import gc
from PIL import Image

# Import from our custom modules
from preprocessing import SeismicDataGenerator, check_dataset_integrity
from metrics import R2Score, MAEMetric, MSEMetric

class Sampling(layers.Layer):
    """Reparameterization trick for VAE"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class SpectrogramVAE(keras.Model):
    def __init__(self, latent_dim=256, image_size=(128, 512, 3), spectra_dim=50, 
                 beta=0.01, **kwargs):
        """
        Variational Autoencoder for seismic spectrograms
        
        Args:
            latent_dim: Dimension of the latent space
            image_size: Input image size (height, width, channels)
            spectra_dim: Dimension of the response spectra
            beta: Weight for KL divergence loss (beta-VAE)
        """
        super(SpectrogramVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.spectra_dim = spectra_dim
        self.beta = beta
        
        # Build the encoder
        self.encoder = self._build_encoder()
        
        # Build the decoder (for spectra prediction)
        self.decoder = self._build_decoder()
        
        # Sample layer
        self.sampling = Sampling()
        
        # Total loss tracker
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.r2_metric = R2Score(name="r2_score")
        self.mae_metric = MAEMetric(name="mae")
        self.mse_metric = MSEMetric(name="mse")
    
    def _build_encoder(self):
        """Build the encoder network"""
        encoder_inputs = keras.Input(shape=self.image_size, name="input_image")
        
        # Convolutional layers
        x = layers.Conv2D(32, 3, strides=2, padding="same")(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(512, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)  # Higher dropout for better generalization
        
        # Output layers for mean and log variance
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        
        # Add cliping to z_log_var to prevent extreme values
        z_log_var = tf.clip_by_value(z_log_var, -20.0, 20.0)
        
        # Define encoder model
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
        return encoder
    
    def _build_decoder(self):
        """Build the decoder network using LSTM for spectra prediction"""
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="z_sampling")
        
        # Reshape for LSTM input - LSTM expects [batch, timesteps, features]
        # We'll interpret the latent vector as a sequence
        # Reshape to (batch_size, latent_dim/feature_size, feature_size)
        feature_size = 32  # Number of features per timestep
        timesteps = self.latent_dim // feature_size
        
        # Ensure we can reshape properly
        if self.latent_dim % feature_size != 0:
            x = layers.Dense(timesteps * feature_size)(latent_inputs)
        else:
            x = latent_inputs
            
        # Reshape to [batch, timesteps, features]
        x = layers.Reshape((timesteps, feature_size))(x)
        
        # LSTM layers
        x = layers.LSTM(512, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.LSTM(256, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Last LSTM layer with return_sequences=False to get a single output vector
        x = layers.LSTM(128, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Final dense layers
        x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer for spectra prediction
        spectra_output = layers.Dense(self.spectra_dim, activation="sigmoid", 
                                    name="output_spectra")(x)
        
        # Define decoder model
        decoder = keras.Model(latent_inputs, spectra_output, name="decoder")
        return decoder
    
    def call(self, inputs):
        """Forward pass"""
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        spectra = self.decoder(z)
        
        return {"output_spectra": spectra, "z_mean": z_mean, "z_log_var": z_log_var}
    
    def train_step(self, data):
        """Custom training step for VAE with NaN protection"""
        inputs, targets = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self(inputs, training=True)
            
            # Compute reconstruction loss
            spectra_loss = tf.reduce_mean(
                keras.losses.mean_absolute_error(
                    targets["output_spectra"], outputs["output_spectra"]
                )
            )
            
            # Making sure z_log_var doesn't have extreme values
            z_log_var_clipped = tf.clip_by_value(outputs["z_log_var"], -10.0, 10.0)
            
            # Compute KL divergence with numerical stability
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var_clipped - tf.square(outputs["z_mean"]) 
                    - tf.exp(z_log_var_clipped),
                    axis=1
                )
            )

            epsilon = 1e-8
            
            # Ensure losses are finite
            spectra_loss = tf.where(tf.math.is_finite(spectra_loss), spectra_loss, tf.constant(epsilon, dtype=spectra_loss.dtype))
            kl_loss = tf.where(tf.math.is_finite(kl_loss), kl_loss, tf.constant(epsilon, dtype=kl_loss.dtype))
            
            # Total loss
            total_loss = spectra_loss + self.beta * kl_loss
            
            # Check for NaN in loss 
            nan_or_inf_detected = tf.logical_or(
                tf.math.is_nan(total_loss), 
                tf.math.is_inf(total_loss)
            )
            
            total_loss = tf.cond(
                nan_or_inf_detected,
                lambda: tf.constant(1.0, dtype=total_loss.dtype),  
                lambda: total_loss  
            )
            
            if tf.executing_eagerly() and nan_or_inf_detected:
                print("WARNING: NaN or Inf detected in loss! Using fallback loss.")
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        
        valid_gradients = []
        for grad in grads:
            if grad is not None:
                # Use tf.where with is_finite check
                valid_grad = tf.where(
                    tf.math.is_finite(grad),
                    grad,
                    tf.zeros_like(grad)
                )
                valid_gradients.append(valid_grad)
            else:
                valid_gradients.append(grad)
        
        self.optimizer.apply_gradients(zip(valid_gradients, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(spectra_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.r2_metric.update_state(targets["output_spectra"], outputs["output_spectra"])
        self.mae_metric.update_state(targets["output_spectra"], outputs["output_spectra"])
        self.mse_metric.update_state(targets["output_spectra"], outputs["output_spectra"])
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "r2_score": self.r2_metric.result(),
            "mae": self.mae_metric.result(),
            "mse": self.mse_metric.result(),
        }

    def test_step(self, data):
        """Custom test step for VAE to track metrics during validation"""
        inputs, targets = data
        
        # Forward pass
        outputs = self(inputs, training=False)
        
        # Compute reconstruction loss
        spectra_loss = tf.reduce_mean(
            keras.losses.mean_squared_error(
                targets["output_spectra"], outputs["output_spectra"]
            )
        )
        
        # Making sure z_log_var doesn't have extreme values
        z_log_var_clipped = tf.clip_by_value(outputs["z_log_var"], -10.0, 10.0)
        
        # Compute KL divergence with numerical stability
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var_clipped - tf.square(outputs["z_mean"]) 
                - tf.exp(z_log_var_clipped),
                axis=1
            )
        )
        
        spectra_loss = tf.where(tf.math.is_finite(spectra_loss), spectra_loss, tf.constant(0.0, dtype=spectra_loss.dtype))
        kl_loss = tf.where(tf.math.is_finite(kl_loss), kl_loss, tf.constant(0.0, dtype=kl_loss.dtype))
        
        total_loss = spectra_loss + self.beta * kl_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(spectra_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.r2_metric.update_state(targets["output_spectra"], outputs["output_spectra"])
        self.mae_metric.update_state(targets["output_spectra"], outputs["output_spectra"])
        self.mse_metric.update_state(targets["output_spectra"], outputs["output_spectra"])
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "r2_score": self.r2_metric.result(),
            "mae": self.mae_metric.result(),
            "mse": self.mse_metric.result(),
        }
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.r2_metric,
            self.mae_metric,
            self.mse_metric,
        ]

# Training function
def train_vae(hdf5_path, images_folder, base_folder, model_folder, history_folder, evaluation_folder,  
              trim_time="3s", batch_size=16, epochs=15, latent_dim=256):
    """
    Train the VAE model
    
    Args:
        hdf5_path: Path to the HDF5 file
        images_folder: Folder containing spectrograms
        base_folder: Base directory for all output
        model_folder: Folder to save the model
        history_folder: Folder to save training history
        evaluation_folder: Folder to save evaluation results
        trim_time: Trim time to use (e.g., "2s", "3s", "25s")
    
    Returns:
        vae: Trained VAE model
        history: Training history
        test_indices: Indices for the test set (for later evaluation)
    """
    image_size = (128, 512)  # Resize spectrograms to this size
    
    # Get the number of samples and response spectra dimension
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = len(f['image_metadata'])
        spectra_dim = len(f['osc_periods'])
        
        print("\nExamining HDF5 metadata:")
        metadata = f['image_metadata'][:]
        for i in range(min(3, len(metadata))):
            event_id = metadata[i][0].decode('utf-8')
            print(f"Event {i} ({event_id}) metadata:")
            for j, field_name in enumerate(metadata.dtype.names):
                try:
                    value = metadata[i][j]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    print(f"  {field_name}: {value}")
                except Exception as e:
                    print(f"  {field_name}: Error decoding - {str(e)}")
    
    # Split data into train, validation, and test sets (75/15/10 split)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_end = int(0.75 * total_samples)
    val_end = int(0.9 * total_samples)  
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]  
    
    print(f"Training on {len(train_indices)} samples ({len(train_indices)/total_samples:.1%})")
    print(f"Validating on {len(val_indices)} samples ({len(val_indices)/total_samples:.1%})")
    print(f"Testing on {len(test_indices)} samples ({len(test_indices)/total_samples:.1%})")
    print(f"Using trim time: {trim_time}")
    
    # Save test indices for later evaluation
    test_indices_path = os.path.join(evaluation_folder, f"test_indices_{trim_time}_lstm.npy")
    np.save(test_indices_path, test_indices)
    print(f"Test indices saved to {test_indices_path}")
    
    # Create data generators
    train_generator = SeismicDataGenerator(
        hdf5_path=hdf5_path,
        images_folder=images_folder,
        batch_size=batch_size,
        trim_time=trim_time,
        indices=train_indices,
        shuffle=True,
        image_size=image_size
    )
    
    val_generator = SeismicDataGenerator(
        hdf5_path=hdf5_path,
        images_folder=images_folder,
        batch_size=batch_size,
        trim_time=trim_time,
        indices=val_indices,
        shuffle=False,
        image_size=image_size
    )
    
    # Create and compile the VAE model
    vae = SpectrogramVAE(
        latent_dim=latent_dim,
        image_size=(*image_size, 3),  
        spectra_dim=spectra_dim,
        beta=0.1  
    )
    
    # Compile the model
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    
    # Setup callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(base_folder, f"vae_model_best_trimmed_{trim_time}_lstm"),
            save_best_only=True,
            monitor="val_loss",
            save_format="tf"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(base_folder, f"vae_logs_trimmed_{trim_time}_lstm"),
            histogram_freq=1
        )
    ]
    
    # Train the model
    history = vae.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save the model
    vae.save(os.path.join(model_folder, f"vae_model_{trim_time}_lstm"), save_format="tf")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(history_folder, f"vae_history_{trim_time}_lstm.csv"), index=False)
    
    # Plot training history
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(history.history['reconstruction_loss'], label='Train')
    plt.plot(history.history['val_reconstruction_loss'], label='Validation')
    plt.title('Reconstruction Loss')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(history.history['kl_loss'], label='Train')
    plt.plot(history.history['val_kl_loss'], label='Validation')
    plt.title('KL Divergence Loss')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(history.history['r2_score'], label='Train')
    plt.plot(history.history['val_r2_score'], label='Validation')
    plt.title('R² Score')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.title('Mean Absolute Error')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(history.history['mse'], label='Train')
    plt.plot(history.history['val_mse'], label='Validation')
    plt.title('Mean Squared Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(history_folder, f"vae_training_history_{trim_time}_lstm.png"))
    plt.close()
    
    return vae, history, test_indices

# Evaluation function
def evaluate_vae(vae, hdf5_path, images_folder, evaluation_folder, trim_time="3s", num_plot_samples=5, 
                 eval_indices=None):
    """
    Evaluate the VAE by calculating metrics on test samples but visualizing only a few random ones
    
    Args:
        vae: Trained VAE model
        hdf5_path: Path to the HDF5 file
        images_folder: Folder containing spectrograms
        evaluation_folder: Folder to save evaluation results
        trim_time: Trim time to use (e.g., "2s", "3s", "25s")
        num_plot_samples: Number of random samples to visualize (default: 5)
        eval_indices: Specific indices to use for evaluation (e.g., test set). If None, random samples will be used.
    
    Returns:
        evaluation_metrics: Dictionary containing evaluation metrics
    """
    # Use the specified trim time for the folder name
    image_type = f"trimmed_{trim_time}"
    
    # Load samples for evaluation
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = len(f['image_metadata'])
        osc_periods = f['osc_periods'][:]
        
        print(f"Total samples available: {total_samples}")
        
        # Determine indices to use for evaluation
        if eval_indices is None:
            num_eval_samples = int(0.1 * total_samples)
            eval_indices = np.random.choice(total_samples, num_eval_samples, replace=False)
            print(f"Using {num_eval_samples} random samples for evaluation (10% of total)")
        else:
            print(f"Using {len(eval_indices)} provided test indices for evaluation")
        
        # Select a subset of indices for visualization
        if num_plot_samples > len(eval_indices):
            num_plot_samples = len(eval_indices)
        plot_indices = np.random.choice(eval_indices, num_plot_samples, replace=False)
        
        print(f"Will visualize {num_plot_samples} random samples from evaluation set")
        
        # Initialize for all metrics
        all_true_norm = []
        all_pred_norm = []
        
        # Store individual metrics for each plotted sample
        individual_metrics = []
        
        # Create the figure for plots
        plt.figure(figsize=(18, 6 * num_plot_samples)) # 15,5 18,6
        gs = gridspec.GridSpec(num_plot_samples * 2, 2, height_ratios=[1, 1] * num_plot_samples, hspace=0.4)
        
        # Process all evaluation samples
        for i, idx in enumerate(eval_indices):
            event_id = f['image_metadata'][idx][0].decode('utf-8')
            
            img_path = os.path.join(images_folder, image_type, f"{event_id}.png")
            
            # Check if image exists
            if not os.path.exists(img_path):
                if i < 10:  
                    print(f"Trying to find image for event {event_id}...")
                
                possible_paths = [
                    img_path,
                    img_path[:-4],
                    os.path.join(images_folder, image_type, f"{event_id}"),
                    os.path.join(images_folder, image_type, f"{event_id}.png"),
                    os.path.join(images_folder, image_type, f"event_{event_id.split('_')[1]}.png"),
                    os.path.join(images_folder, image_type, f"event_{event_id.split('_')[1].lstrip('0')}.png"),
                ]
                
                found_path = False
                for path in possible_paths:
                    if os.path.exists(path):
                        img_path = path
                        found_path = True
                        if i < 10:  
                            print(f"Found image for event {event_id} at: {path}")
                        break
                
                if not found_path:
                    if i < 10: 
                        print(f"WARNING: Cannot find image for event {event_id}. Skipping.")
                    continue
            
            try:
                img = Image.open(img_path)
                
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                img = img.resize((512, 128))  # Match the model's input size
                img_array = np.array(img) / 255.0
                
                # Get ground truth spectra
                true_spectra_log = f[f"{event_id}/response_spectra/spec_accel"][:]
        
                # Apply normalization
                true_min = np.min(true_spectra_log)
                true_max = np.max(true_spectra_log)
                
                # Handle edge case where min equals max
                if true_max - true_min > 1e-8:
                    true_spectra_norm = (true_spectra_log - true_min) / (true_max - true_min)
                else:
                    true_spectra_norm = true_spectra_log  # No normalization if range is tiny
                
                # Predict with VAE
                prediction = vae.predict({"input_image": np.expand_dims(img_array, 0)}, verbose=0)
                pred_spectra_norm = prediction["output_spectra"][0]
                
                # Store normalized values for global metrics calculation
                all_true_norm.append(true_spectra_norm)
                all_pred_norm.append(pred_spectra_norm)
                
                # Plot only if this sample is in the plotting subset
                if idx in plot_indices:
                    plot_idx = np.where(plot_indices == idx)[0][0]  # Get the index in plot_indices
                    
                    # Denormalize prediction for visualization
                    if true_max - true_min > 1e-8:
                        pred_spectra_log = pred_spectra_norm * (true_max - true_min) + true_min
                    else:
                        pred_spectra_log = pred_spectra_norm
                    
                    # Calculate individual metrics for this sample
                    individual_r2 = r2_score(true_spectra_norm, pred_spectra_norm)
                    individual_mae = mean_absolute_error(true_spectra_norm, pred_spectra_norm)
                    individual_mse = mean_squared_error(true_spectra_norm, pred_spectra_norm)
                    individual_rmse = np.sqrt(individual_mse)
                    
                    # Store individual metrics
                    individual_metrics.append({
                        'event_id': event_id,
                        'R2': individual_r2,
                        'MAE': individual_mae,
                        'MSE': individual_mse
                    })
                    
                    # Get PGA from metadata if available
                    pga_value = None
                    try:
                        # Get the metadata from HDF5
                        metadata = f[f"{event_id}/metadata"]
                        full_trace = f[f"{event_id}/full_trace"][:]
                        sampling_rate = 100
                        time_axis = np.linspace(0, len(full_trace)/sampling_rate, len(full_trace))

                        # Get PGA from metadata attributes
                        pga_value = np.log(metadata.attrs['PGA (m/s2)'])
                        pga_label = f'PGA (log): {pga_value:.4f}'

                        # Get station's informations
                        magnitude = metadata.attrs['Magnitude']
                        mag_type = metadata.attrs['Magnitude_type']
                        epicenter = metadata.attrs['Epicenter (km)']
                    except Exception as e:
                        print(f"Could not get PGA from metadata for event {event_id}: {str(e)}")
                    
                    row_start = plot_idx * 2

                    # Plot full waveform
                    ax1 = plt.subplot(gs[row_start, 0])
                    ax1.plot(time_axis, full_trace, 'k', linewidth=1.0)
                    ax1.set_title(f"Full Trace - {event_id} | {magnitude} {mag_type} | Ep: {epicenter} km ")
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Acceleration (m/s²)')
                    ax1.grid(True, which="both", ls="-", alpha=0.2)

                    # Plot response spectra in right side (spanning two rows)
                    ax2 = plt.subplot(gs[row_start:row_start+2, 1])
                    ax2.plot(osc_periods, true_spectra_log, 'b', linewidth=2.0, label='True Spectra (log)')
                    ax2.plot(osc_periods, pred_spectra_log, 'r', linewidth=2.0, label='Recon Spectra (log)')
                    
                    # Add metrics as text box
                    metrics_text = f"R²: {individual_r2:.4f}\nMAE: {individual_mae:.4f}\nMSE: {individual_mse:.4f}\n RMSE: {individual_rmse:.4f}"
                    ax2.text(0.15, 0.97, metrics_text, transform=plt.gca().transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Add PGA as horizontal line if available
                    if pga_value is not None:
                        ax2.axhline(y=pga_value, color='g', linestyle='--', label=pga_label)
                    
                    ax2.set_xlabel('Period (s)')
                    ax2.set_ylabel('log(Spectral Acceleration) (g)')
                    ax2.set_title(f"Response Spectra - {event_id} | {magnitude} {mag_type} | Ep: {epicenter} km")
                    ax2.grid(True, which="both", ls="-", alpha=0.2)
                    ax2.legend()

                    # Plot spectrogram in bottom-left
                    ax3 = plt.subplot(gs[row_start+1, 0])
                    ax3.imshow(img_array)
                    ax3.set_title(f"Trimmed ({trim_time}) Spectrogram - {event_id} | {magnitude} {mag_type} | Ep: {epicenter} km")
                    ax3.axis('off')
                
            except Exception as e:
                if i < 10:  # Only print for the first few
                    print(f"Error processing sample {idx} (event {event_id}): {str(e)}")
                continue
        
        # Calculate metrics on all successfully processed samples
        if all_true_norm and all_pred_norm:
            try:
                print(f"Calculating metrics on {len(all_true_norm)} successfully processed samples")
                all_true_flat = np.array(all_true_norm).flatten()
                all_pred_flat = np.array(all_pred_norm).flatten()

                r2 = r2_score(all_true_flat, all_pred_flat)
                mae_overall = mean_absolute_error(all_true_flat, all_pred_flat)
                mse_overall = mean_squared_error(all_true_flat, all_pred_flat)
                rmse_overall = np.sqrt(mse_overall)

                evaluation_metrics = {
                    'Overall R2 (normalized)': r2,
                    'Overall MAE (normalized)': mae_overall,
                    'Overall MSE (normalized)': mse_overall,
                    'Overall RMSE (normalized)': rmse_overall,
                    'Number of samples evaluated': len(all_true_norm)
                }

                # Add information about the data split to the title
                title_suffix = "Test Set" if eval_indices is not None else "Random Samples"
                plt.suptitle(f"VAE Predictions ({trim_time} Trimmed) - {title_suffix}\nR² Score: {r2:.4f}, MAE: {mae_overall:.4f}, MSE: {mse_overall:.4f}, RMSE: {rmse_overall:.4f}\n (Calculated on {len(all_true_norm)} samples)", fontsize=16)
                print(f"Overall R² Score: {r2:.4f}, MAE: {mae_overall:.4f}, MSE: {mse_overall:.4f}, RMSE: {rmse_overall:.4f}")
                
                # Save metrics to Excel
                overall_df = pd.DataFrame({
                    'Metric': ['R2', 'MAE', 'MSE', 'RMSE', 'Number of samples'],
                    'Normalized': [r2, mae_overall, mse_overall, rmse_overall, len(all_true_norm)],
                })
                
                # Create DataFrame for individual metrics
                individual_df = pd.DataFrame(individual_metrics)

                excel_path = os.path.join(evaluation_folder, f'vae_evaluation_{trim_time}_lstm.xlsx')
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    overall_df.to_excel(writer, sheet_name='Overall Metrics', index=False)
                    if len(individual_metrics) > 0:
                        individual_df.to_excel(writer, sheet_name='Individual Metrics', index=False)
                
                print(f"Evaluation metrics saved to {excel_path}")

            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")
                evaluation_metrics = {'error': str(e)}
        else:
            print("WARNING: No valid samples were found for evaluation. Check the path structure.")
            evaluation_metrics = {'error': 'No valid samples found'}
        
        plt.tight_layout()
        plt.subplots_adjust(top = 0.93, hspace=0.4, wspace=0.2) # Make room for suptitle
        plt.savefig(os.path.join(evaluation_folder, f"vae_predictions_{trim_time}_lstm.png"))
        plt.close()

        return evaluation_metrics