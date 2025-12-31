import torch
import torch.nn as nn
import torch.optim as optim
import gudhi as gd
import numpy as np
from sklearn.decomposition import PCA
from collections import deque
import copy
import time # For timing operations

# ----------------------------------------------------------
# 辅助函数 - 拓扑特征计算
# ----------------------------------------------------------

class TopologyFeatureExtractor:
    def __init__(self,
                 n_pca_components=10,       # PCA 降维维度
                 max_history_points=100,    # 缓存多少个模型参数点用于 PH
                 persistence_threshold_percentile=10, # 过滤掉生命周期最长的 N% 的同调特征
                 r_max_for_rips_quantile=0.95, # Rips 复形的最大半径，设为 pairwise 距离的 N% 分位数
                 use_hessian_for_morse=False, # 是否尝试使用海森矩阵计算 Morse 指标 (高级且计算昂贵)
                 simple_morse_threshold_grad_norm=1e-5 # 梯度范数阈值，用于简化版 Morse 计算
                ):
        self.n_pca_components = n_pca_components
        self.max_history_points = max_history_points
        self.persistence_threshold_percentile = persistence_threshold_percentile
        self.r_max_for_rips_quantile = r_max_for_rips_quantile
        self.use_hessian_for_morse = use_hessian_for_morse
        self.simple_morse_threshold_grad_norm = simple_morse_threshold_grad_norm

        self.history_params = deque(maxlen=max_history_points)
        self.pca = PCA(n_components=n_pca_components)
        self.fitted_pca = False
        self._pca_fit_needed = True # Flag to indicate if PCA needs refitting

    def _flatten_params(self, model):
        """展平模型所有参数，并返回一个 numpy 数组"""
        flat_params = []
        for p in model.parameters():
            if p.requires_grad:
                # Ensure data is on CPU for numpy conversion
                flat_params.append(p.data.cpu().numpy().flatten())
        if not flat_params:
            return np.array([])
        return np.concatenate(flat_params)

    def _estimate_r_max(self, low_dim_points):
        """根据点集估计 Rips 复形的 max_edge_length"""
        if low_dim_points.shape[0] < 2:
            return 0.1 # Default small value
        try:
            from scipy.spatial.distance import pdist
            distances = pdist(low_dim_points)
            if len(distances) == 0:
                return 0.1
            # Use specified quantile of pairwise distances
            return np.percentile(distances, self.r_max_for_rips_quantile * 100)
        except Exception as e:
            print(f"Warning: Error estimating r_max: {e}. Using default 0.1.")
            return 0.1

    def _compute_betti_numbers(self, low_dim_params):
        """
        使用持久同调计算 Betti 数。
        low_dim_params: 形状为 (N, D_lowdim) 的 numpy 数组。
        """
        if low_dim_params.shape[0] < 2: # Need at least two points
            return {'b0': 1, 'b1': 0, 'b2': 0}

        # Estimate r_max if not explicitly set
        current_r_max = self.r_max_for_rips if hasattr(self, 'r_max_for_rips') and self.r_max_for_rips is not None else None
        if current_r_max is None:
            current_r_max = self._estimate_r_max(low_dim_params)

        try:
            rips_complex = gd.RipsComplex(points=low_dim_params, max_edge_length=current_r_max)
            # Compute up to dimension 2 (Betti-0, Betti-1, Betti-2)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

            # Calculate all persistence intervals first
            all_persistence = simplex_tree.persistence()
            if not all_persistence:
                 return {'b0': 1, 'b1': 0, 'b2': 0}

            # Filter based on dynamic persistence threshold
            lifetimes = [death - birth for dim, (birth, death) in all_persistence]
            if not lifetimes:
                return {'b0': 1, 'b1': 0, 'b2': 0}

            # Calculate dynamic threshold using percentile
            # Remove features with lifetime shorter than this threshold
            # E.g., if percentile is 10, we remove the shortest 10% of features
            dynamic_persistence_threshold = np.percentile(lifetimes, self.persistence_threshold_percentile)

            # Ensure threshold is not zero or negative if possible
            dynamic_persistence_threshold = max(dynamic_persistence_threshold, 1e-6)

            filtered_persistence = [(dim, interval) for dim, interval in all_persistence
                                    if interval[1] - interval[0] >= dynamic_persistence_threshold]

            # Count Betti numbers from filtered persistence intervals
            b0, b1, b2 = 0, 0, 0
            for dim, (birth, death) in filtered_persistence:
                if dim == 0: b0 += 1
                elif dim == 1: b1 += 1
                elif dim == 2: b2 += 1

            # Ensure at least one connected component (B0 >= 1)
            if b0 == 0: b0 = 1

            # print(f"DEBUG: Betti Numbers (threshold={dynamic_persistence_threshold:.4f}): b0={b0}, b1={b1}, b2={b2}") # For debugging
            return {'b0': b0, 'b1': b1, 'b2': b2}

        except Exception as e:
            print(f"Warning: Error computing Betti numbers: {e}. Returning default values.")
            return {'b0': 1, 'b1': 0, 'b2': 0}

    def _compute_morse_indicator(self, model, loss_fn, optimizer_state, data_loader=None, sample_data=None):
        """
        计算 Morse 指标。
        - `use_hessian_for_morse=True`: 尝试使用海森矩阵（近似）。这是计算密集型且复杂的。
        - `use_hessian_for_morse=False`: 使用简化的梯度范数分级。
        """
        if not self.use_hessian_for_morse:
            # Simplified Morse Indicator based on gradient norm
            # Mapping: 0 (min/flat), 1 (saddle), 2 (descending)
            grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])).item()
            if grad_norm < self.simple_morse_threshold_grad_norm:
                return 0 # Potentially minimum or flat region
            elif grad_norm < self.simple_morse_threshold_grad_norm * 100: # Heuristic threshold
                return 1 # Potentially saddle region
            else:
                return 2 # Likely descending
        else:
            # --- Placeholder for Complex Hessian-based Morse Indicator ---
            # This part requires significant implementation effort (e.g., K-FAC, Randomized Hessian)
            # For now, we'll print a warning and return a fallback.
            print("Warning: use_hessian_for_morse=True, but Hessian computation is not implemented.")
            print("         Returning default Morse indicator value of 1. Implement Hessian logic for full functionality.")
            # Example of what could be done with eigenvalues of Hessian:
            # try:
            #     # Assume `compute_approx_hessian_eigenvalues` is implemented and returns a list/tensor of eigenvalues
            #     eigenvalues = compute_approx_hessian_eigenvalues(model, loss_fn, sample_data)
            #     num_negative_eigenvalues = torch.sum(torch.tensor(eigenvalues) < -1e-6).item()
            #     return num_negative_eigenvalues # Morse indicator = number of negative eigenvalues
            # except Exception as e:
            #     print(f"Error during approximate Hessian computation: {e}")
            #     return 1 # Fallback value
            return 1 # Fallback value

    def update_history(self, model, current_iter):
        """
        将当前模型参数添加到历史记录中。
        current_iter is passed for potential future intelligent sampling strategies.
        """
        param_state = self._flatten_params(model)
        if param_state.size > 0: # Only add if parameters exist
            self.history_params.append(param_state)
            self._pca_fit_needed = True # Mark PCA as needing refitting

    def fit_pca_if_needed(self):
        """拟合 PCA 模型，如果历史点足够多且 PCA 未拟合"""
        if self._pca_fit_needed and len(self.history_params) >= self.max_history_points and len(self.history_params) >= self.n_pca_components + 1:
            history_array = np.array(list(self.history_params))
            if history_array.shape[0] > 0 and history_array.shape[1] > 0:
                try:
                    self.pca.fit(history_array)
                    self.fitted_pca = True
                    self._pca_fit_needed = False
                    # print("DEBUG: PCA fitted with historical parameters.")
                except Exception as e:
                    print(f"Warning: Error fitting PCA: {e}")
                    self.fitted_pca = False
                    self._pca_fit_needed = True # Allow retry
            else:
                # print("DEBUG: History points empty or invalid for PCA fit.")
                self.fitted_pca = False # Cannot fit if no valid data
                self._pca_fit_needed = True


    def get_topology_features(self, model, loss_fn, optimizer_state, data_loader=None, sample_data=None):
        """计算并返回 Betti 数和 Morse 指标"""

        # Ensure PCA is fitted if needed
        self.fit_pca_if_needed()

        # 1. Compute Betti Numbers
        betti_numbers = {'b0': 1, 'b1': 0, 'b2': 0}
        if self.fitted_pca and len(self.history_params) >= 2: # PCA must be fitted, and at least two points available
            try:
                history_array = np.array(list(self.history_params))
                # Use fitted PCA for dimensionality reduction
                # Ensure the dimensionality of history_array matches PCA's expected input if fit was partial
                low_dim_params = self.pca.transform(history_array)
                betti_numbers = self._compute_betti_numbers(low_dim_params)
            except Exception as e:
                print(f"Warning: Error in Betti number calculation after PCA transform: {e}")
                # Fallback to default

        # 2. Compute Morse Indicator
        # This is often more expensive. It might be called less frequently.
        # For now, we compute it every time get_topology_features is called.
        morse_indicator = 1 # Default value
        try:
            morse_indicator = self._compute_morse_indicator(model, loss_fn, optimizer_state, data_loader, sample_data)
        except Exception as e:
            print(f"Warning: Error during Morse indicator calculation: {e}")
            morse_indicator = 1 # Fallback

        return betti_numbers, morse_indicator

# ----------------------------------------------------------
# 拓扑引导的 Adam 优化器
# ----------------------------------------------------------

class TopologyGuidedAdam(optim.Adam):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 # --- Topology Guidance Hyperparameters ---
                 # Thresholds for Betti numbers and Morse indicator to trigger parameter adjustments
                 # These values are heuristic and need to be tuned.
                 betti_thresholds={'b0': 5, 'b1': 3, 'b2': 1},
                 morse_threshold=1, # For simplified Morse indicator (0=min/flat, 1=saddle, 2=descending)

                 # --- TopologyFeatureExtractor Configuration ---
                 n_pca_components=10,
                 max_history_points=100,
                 persistence_threshold_percentile=10,
                 r_max_for_rips_quantile=0.95,
                 use_hessian_for_morse=False, # Set to True for advanced (but unimplemented) Hessian calculation
                 simple_morse_threshold_grad_norm=1e-5,

                 # --- Update Frequency Control ---
                 # How often to re-calculate topology features and apply guidance.
                 # Set to a high value (e.g., 1e9) to disable dynamic updates and use fixed thresholds.
                 topology_update_freq_iter=100, # Recalculate every N iterations
                 topology_update_freq_epoch=1,  # Recalculate every N epochs

                 # --- Guidance Strength (Experimental) ---
                 # Multipliers to adjust the impact of topology features on Adam params
                 beta1_reduction_factor_high_complexity=0.8, # Factor to reduce beta1 when complexity is high
                 beta1_increase_factor_low_complexity=1.05,  # Factor to increase beta1 when complexity is low
                 eps_reduction_factor_high_morse=10.0,      # Factor to reduce eps when morse indicator is high
                 eps_increase_factor_low_morse=10.0         # Factor to increase eps when morse indicator is low
                ):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

        self.betti_thresholds = betti_thresholds
        self.morse_threshold = morse_threshold

        self.topology_extractor = TopologyFeatureExtractor(
            n_pca_components=n_pca_components,
            max_history_points=max_history_points,
            persistence_threshold_percentile=persistence_threshold_percentile,
            r_max_for_rips_quantile=r_max_for_rips_quantile,
            use_hessian_for_morse=use_hessian_for_morse,
            simple_morse_threshold_grad_norm=simple_morse_threshold_grad_norm
        )

        # State tracking for update frequencies
        self.current_iter = 0
        self.current_epoch = 0
        self.topology_update_freq_iter = topology_update_freq_iter
        self.topology_update_freq_epoch = topology_update_freq_epoch
        self.last_topology_update_iter = -self.topology_update_freq_iter # To trigger on first valid iteration/epoch
        self.last_topology_update_epoch = -self.topology_update_freq_epoch # To trigger on first valid iteration/epoch

        # Cached topology features
        self.cached_betti_numbers = {'b0': 1, 'b1': 0, 'b2': 0}
        self.cached_morse_indicator = 1
        self._topology_features_calculated_at_least_once = False

        # Store original Adam hyperparameters for resetting and scaling
        self.original_betas = list(betas)
        self.original_eps = eps

        # Store guidance strength factors
        self.beta1_reduction_factor_high_complexity = beta1_reduction_factor_high_complexity
        self.beta1_increase_factor_low_complexity = beta1_increase_factor_low_complexity
        self.eps_reduction_factor_high_morse = eps_reduction_factor_high_morse
        self.eps_increase_factor_low_morse = eps_increase_factor_low_morse

        # Context for topology feature calculation (model, loss_fn, etc.)
        self._context = {
            'model': None, 'loss_fn': None, 'optimizer_state': None,
            'data_loader': None, 'sample_data': None, 'current_loss': None
        }

    def set_context(self, model, loss_fn, data_loader=None, sample_data=None):
        """
        Sets the necessary context for topology feature calculation.
        This method MUST be called before the first optimizer.step() or periodically
        if model, loss_fn, data_loader, or sample_data change during training.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("`model` must be a torch.nn.Module.")
        if not callable(loss_fn):
            raise TypeError("`loss_fn` must be a callable function.")

        self._context['model'] = model
        self._context['loss_fn'] = loss_fn
        self._context['optimizer_state'] = self.state # Optimizer's internal state
        self._context['data_loader'] = data_loader
        self._context['sample_data'] = sample_data
        # print("DEBUG: Topology context set.")

    def increment_iter(self):
        """Call this at the end of each training iteration (batch)."""
        self.current_iter += 1

    def set_epoch(self, epoch):
        """Call this at the beginning of each training epoch."""
        self.current_epoch = epoch
        # Reset last epoch update tracker when a new epoch starts
        # This allows epoch-based frequency control to function correctly for the new epoch
        # We might want to trigger an update at the start of a new epoch if freq=1
        self.last_topology_update_epoch = epoch - 1 # Ensure first epoch triggers if freq=1

    def _check_topology_update(self):
        """Determines if topology features need to be recalculated based on frequency controls."""
        if self.topology_update_freq_iter <= 0 and self.topology_update_freq_epoch <= 0:
            return False # Updates are disabled

        # Check iteration frequency
        trigger_iter = (self.current_iter > self.last_topology_update_iter + self.topology_update_freq_iter - 1)
        # Check epoch frequency
        trigger_epoch = (self.current_epoch > self.last_topology_update_epoch + self.topology_update_freq_epoch - 1)

        # If epoch frequency is set, it takes precedence or works in conjunction
        if self.topology_update_freq_epoch > 0 and trigger_epoch:
            # If we are within the epoch update interval
            return True
        elif self.topology_update_freq_iter > 0 and trigger_iter:
            # If we are within the iteration update interval and epoch interval not met
            return True
        else:
            return False

    def _update_topology_features(self):
        """
        Recalculates and caches topology features.
        Returns True if calculation was successful, False otherwise.
        """
        if self._context['model'] is None or self._context['loss_fn'] is None:
            # print("Warning: Topology context (model, loss_fn) not set. Skipping topology feature update.")
            return False

        model = self._context['model']
        loss_fn = self._context['loss_fn']
        optimizer_state = self._context['optimizer_state']
        data_loader = self._context['data_loader']
        sample_data = self._context['sample_data']

        # Crucial: Ensure gradients are available for Morse calculation.
        # Betti number calculation relies on cached parameters only.
        # Morse calculation (especially simplified) requires current gradients.
        has_gradients = any(p.grad is not None for p in model.parameters())
        if not has_gradients:
            # print("Warning: No gradients found. Betti numbers will be computed, but Morse indicator might be inaccurate or skipped.")
            # Decide how to handle this: skip or use cached features.
            # For now, we will attempt Betti calculation and warn about Morse.
            pass # Proceed to Betti, but Morse will be a fallback.

        try:
            # Update history with current model parameters (for Betti calculation)
            self.topology_extractor.update_history(model, self.current_iter)

            # Compute features
            betti_numbers, morse_indicator = self.topology_extractor.get_topology_features(
                model, loss_fn, optimizer_state, data_loader, sample_data
            )

            self.cached_betti_numbers = betti_numbers
            self.cached_morse_indicator = morse_indicator
            self._topology_features_calculated_at_least_once = True

            # Update frequency counters
            self.last_topology_update_iter = self.current_iter
            self.last_topology_update_epoch = self.current_epoch

            # print(f"DEBUG: Updated Topology Features: Betti={betti_numbers}, Morse={morse_indicator}")
            return True
        except Exception as e:
            print(f"Error during topology feature calculation: {e}")
            # If calculation fails, we might want to reset to original parameters
            # to avoid applying potentially corrupted guidance.
            self.reset_to_original_params()
            return False

    def _apply_topology_guidance(self):
        """
        Applies Adam hyperparameter adjustments based on cached topology features.
        """
        betti = self.cached_betti_numbers
        morse = self.cached_morse_indicator

        # --- Dynamic Beta1 Adjustment (Momentum) ---
        beta1 = self.original_betas[0] # Start with original beta1

        # Heuristic: Higher complexity generally means lower momentum (more responsive to current gradients)
        # Lower complexity means higher momentum (better exploration of flat regions)
        complexity_score = (betti.get('b0', 1) * 1.5 + # Higher B0 might mean more disconnected flat areas
                            betti.get('b1', 0) * 5.0 + # Higher B1 (loops) usually means complex landscape
                            betti.get('b2', 0) * 10.0) # Higher B2 (voids) also indicates complexity

        # Example mapping:
        # If complexity is high, reduce beta1. If low, increase beta1.
        if complexity_score > sum(self.betti_thresholds.values()) * 1.2: # Significantly complex
            beta1 = max(0.5, self.original_betas[0] * self.beta1_reduction_factor_high_complexity)
        elif complexity_score < sum(self.betti_thresholds.values()) * 0.5: # Significantly simple/flat
            beta1 = min(0.999, self.original_betas[0] * self.beta1_increase_factor_low_complexity)
        else:
            # Intermediate complexity, might use original or slight adjustment based on individual thresholds
            if betti.get('b1', 0) > self.betti_thresholds['b1']:
                beta1 = max(0.6, self.original_betas[0] * 0.9) # Reduce for high loops
            if betti.get('b2', 0) > self.betti_thresholds['b2']:
                beta1 = max(0.6, beta1 * 0.9) # Reduce further for high voids

        # Ensure beta1 is within reasonable bounds
        beta1 = max(0.1, min(0.999, beta1))

        # --- Dynamic Epsilon Adjustment (Stability) ---
        eps = self.original_eps

        # Heuristic: Higher Morse indicator (more saddle points/complex curvature) requires more stability (smaller epsilon)
        # Lower Morse indicator (more minimum/flatness) can tolerate more exploration (larger epsilon)
        if morse > self.morse_threshold:
            eps = max(1e-10, self.original_eps / self.eps_reduction_factor_high_morse) # Reduce eps
        else:
            eps = min(1e-5, self.original_eps * self.eps_increase_factor_low_morse) # Increase eps

        # Ensure eps is within reasonable bounds
        eps = max(1e-10, min(1e-5, eps))

        # --- Apply Adjusted Hyperparameters ---
        # Modify the param_groups directly. Adam's internal update will use these.
        for group in self.param_groups:
            group['betas'] = (beta1, group['betas'][1]) # Only adjust beta1
            group['eps'] = eps

        # print(f"Applied Topology Guidance: New betas={self.param_groups[0]['betas']:.4f}, New eps={self.param_groups[0]['eps']:.1e}")

    def reset_to_original_params(self):
        """Resets optimizer parameters to their initial Adam values."""
        for group in self.param_groups:
            group['betas'] = (self.original_betas[0], self.original_betas[1])
            group['eps'] = self.original_eps
        # print("DEBUG: Resetting Adam parameters to original values.")

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Calculates topology features and applies guidance before the standard Adam step.
        """
        loss = None
        if closure is not None:
            # Execute the closure first to compute gradients.
            # The closure should internally perform forward and backward passes.
            try:
                # loss = closure() # This is how it's typically used
                # If closure computes loss and backward, gradients are populated.
                # We might need to capture the loss value if needed for context.
                loss = closure()
            except Exception as e:
                print(f"Error during closure execution: {e}")
                raise e # Re-raise the exception

        # --- Topology Update and Guidance ---
        # Check if it's time to update topology features
        if self._check_topology_update():
            # Attempt to update topology features
            update_success = self._update_topology_features()

            if update_success:
                # If features were updated successfully, apply guidance
                self._apply_topology_guidance()
            else:
                # If update failed, reset to original params to be safe.
                # print("Topology update failed. Resetting Adam parameters to original.")
                self.reset_to_original_params()
        else:
            # If not due for update, but features have been calculated before,
            # we still apply the guidance based on cached features.
            if self._topology_features_calculated_at_least_once:
                self._apply_topology_guidance()
            else:
                # If features haven't been calculated yet (e.g., early iterations before freq met),
                # ensure we are using original Adam parameters.
                # This is implicitly handled if reset_to_original_params is called on failure,
                # or if _apply_topology_guidance is skipped.
                pass

        # --- Standard Adam Step ---
        # Call the parent class's step method. It will use the currently set param_groups
        # which may have been modified by _apply_topology_guidance.
        # If closure was provided and successful, it might need to be passed again.
        # If closure was not provided, super().step() will use existing gradients.
        super_step_return = super().step(closure)

        return super_step_return # Return value of super().step(), typically None

    def zero_grad(self):
        """Clears gradients of all optimized parameters."""
        super().zero_grad()

# ----------------------------------------------------------
# 使用示例 (示意)
# ----------------------------------------------------------
if __name__ == "__main__":
    # --- 1. Define a simple model ---
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # --- 2. Model, Loss, Data Setup ---
    model = SimpleNet()
    loss_fn = nn.CrossEntropyLoss()
    # Placeholder for data_loader, sample_data. In real training, these would be DataLoaders.
    # We need sample_data for Morse indicator calculation if use_hessian_for_morse is True.
    sample_data_for_morse = torch.randn(32, 10) # A sample batch of data for Hessian approximation
    # In this example, we use simplified Morse, so sample_data is not strictly needed by the extractor.

    # --- 3. Initialize TopologyGuidedAdam ---
    optimizer = TopologyGuidedAdam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,

        # --- Topology Guidance Hyperparameters ---
        betti_thresholds={'b0': 5, 'b1': 3, 'b2': 1}, # Example thresholds
        morse_threshold=1,                          # Example threshold for simplified Morse

        # --- TopologyFeatureExtractor Configuration ---
        n_pca_components=10,
        max_history_points=100,
        persistence_threshold_percentile=10, # Filter shortest 10% life cycles
        r_max_for_rips_quantile=0.95,       # Rips edge max length from 95th percentile dist
        use_hessian_for_morse=False,        # Set to True if you have Hessian logic implemented
        simple_morse_threshold_grad_norm=1e-5, # Threshold for simplified Morse (gradient norm)

        # --- Update Frequency Control ---
        topology_update_freq_iter=50,       # Recalculate features every 50 iterations
        topology_update_freq_epoch=2        # Recalculate features every 2 epochs (if epoch counter is used)
    )

    # --- 4. Set Optimizer Context (Crucial!) ---
    # This provides the optimizer with access to model, loss_fn, etc.
    optimizer.set_context(model, loss_fn, data_loader=None, sample_data=sample_data_for_morse)

    # --- 5. Simulate Training Loop ---
    num_epochs = 10
    steps_per_epoch = 100
    total_iters = num_epochs * steps_per_epoch

    print("Starting simulated training loop...")

    for epoch in range(num_epochs):
        optimizer.set_epoch(epoch) # Inform optimizer about the current epoch
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        for step_in_epoch in range(steps_per_epoch):
            optimizer.increment_iter() # Inform optimizer about the current iteration

            # --- Simulate a Training Step ---
            # 1. Zero gradients
            optimizer.zero_grad()

            # 2. Generate dummy data and labels
            inputs = torch.randn(32, 10)
            labels = torch.randint(0, 2, (32,)) # Binary classification example

            # 3. Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # 4. Backward pass (computes gradients)
            #    This is essential for the optimizer to work.
            loss.backward()

            # 5. Optimizer step (includes topology guidance)
            #    If you were using a closure, it would be passed here: optimizer.step(closure=...)
            optimizer.step()

            # --- Logging ---
            if optimizer.current_iter % 20 == 0:
                print(f"  Iter [{optimizer.current_iter}/{total_iters}] Loss: {loss.item():.4f}")
                # Access current betas and eps from optimizer's param_groups for logging
                current_betas = optimizer.param_groups[0]['betas']
                current_eps = optimizer.param_groups[0]['eps']
                print(f"    Current Adam params: betas={current_betas}, eps={current_eps:.1e}")

        # --- Optional: Epoch End Actions ---
        # Evaluate model, decay learning rate, etc.
        # The topology features will be recalculated if topology_update_freq_epoch is met.

    print("\nSimulated training finished.")
