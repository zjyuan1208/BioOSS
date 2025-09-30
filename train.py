"""
This module defines functions for creating datasets, building models, and training them using JAX
and Equinox. The main function, `create_dataset_model_and_train`, is designed to initialise the
dataset, construct the model, and execute the training process.

The function `create_dataset_model_and_train` takes the following arguments:

- `seed`: A random seed for reproducibility.
- `data_dir`: The directory where the dataset is stored.
- `use_presplit`: A boolean indicating whether to use a pre-split dataset.
- `dataset_name`: The name of the dataset to load and use for training.
- `output_step`: For regression tasks, the number of steps to skip before outputting a prediction.
- `metric`: The metric to use for evaluation. Supported values are `'mse'` for regression and `'accuracy'` for
            classification.
- `include_time`: A boolean indicating whether to include time as a channel in the time series data.
- `T`: The maximum time value to scale time data to [0, T].
- `model_name`: The name of the model architecture to use.
- `stepsize`: The size of the intervals for the Log-ODE method.
- `logsig_depth`: The depth of the Log-ODE method. Currently implemented for depths 1 and 2.
- `model_args`: A dictionary of additional arguments to customise the model.
- `num_steps`: The number of steps to train the model.
- `print_steps`: How often to print the loss during training.
- `lr`: The learning rate for the optimiser.
- `lr_scheduler`: The learning rate scheduler function.
- `batch_size`: The number of samples per batch during training.
- `output_parent_dir`: The parent directory where the training outputs will be saved.

The module also includes the following key functions:

- `calc_output`: Computes the model output, handling stateful and nondeterministic models with JAX's `vmap` for
                 batching.
- `classification_loss`: Computes the loss for classification tasks, including optional regularisation.
- `regression_loss`: Computes the loss for regression tasks, including optional regularisation.
- `make_step`: Performs a single optimisation step, updating model parameters based on the computed gradients.
- `train_model`: Handles the training loop, managing metrics, early stopping, and saving progress at regular intervals.
"""

import os
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pickle

from data_dir.datasets import create_dataset
from models.generate_model import create_model


@eqx.filter_jit
def calc_output(model, X, state, key, stateful, nondeterministic):
    if stateful:
        if nondeterministic:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None, None), out_axes=(0, None)
            )(X, state, key)
        else:
            output, state = jax.vmap(
                model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
            )(X, state)
    elif nondeterministic:
        output = jax.vmap(model, in_axes=(0, None))(X, key)
    else:
        output = jax.vmap(model)(X)

    return output, state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    norm = 0
    if hasattr(model, 'lip2') and model.lip2:
        for layer in model.vf.mlp.layers:
            norm += jnp.mean(
                jnp.linalg.norm(layer.weight, axis=-1)
                + jnp.linalg.norm(layer.bias, axis=-1)
            )
        norm *= model.lambd

    return (
        jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1)) + norm,
        state,
    )

# @eqx.filter_jit
# @eqx.filter_value_and_grad(has_aux=True)
# def simple_predictive_loss(model, x, state, key):
#     """
#     A simplified predictive coding loss that uses calc_output.
#     """
#     # Split key for different forward passes
#     key1, key2 = jr.split(key)
#
#     # Use calc_output for both predictions to ensure proper data handling
#     pred_y1, _ = calc_output(model, x, state, key1, False, False)
#
#     # Simple prediction: shift input data rather than direct next-step prediction
#     if x.shape[0] > 1:  # Only if we have more than one timestep
#         # Take time slices of the data
#         n_samples = x.shape[0]
#
#         # Use first n-1 samples as input
#         x_prev = x[:-1]
#         # Use last n-1 samples as target
#         target = x[1:]
#
#         # Calculate predictions for both segments
#         pred_prev, _ = calc_output(model, x_prev, state, key2, False, False)
#
#         # Simple MSE between consecutive predictions
#         pred_loss = jnp.mean((pred_prev - target[:, 0]) ** 2)  # Assuming target shape is appropriate
#     else:
#         # Not enough timesteps for prediction, use dummy loss
#         pred_loss = jnp.array(0.0)
#
#     return pred_loss, {'pred_loss': pred_loss}


# def predictive_coding_loss(model, x, state, key, prediction_horizon=1):
#     """
#     Implements a predictive coding loss where each neuron predicts its state
#     at the next time step, following the neuronal least-action principle.
#     """
#     # Run the model to get intermediate activations (without final output layer)
#     dropkeys = jr.split(key, len(model.blocks))
#
#     # Check and prepare input shape
#     # If x is (batch, seq_len, features), we need to ensure it matches the encoder's expectations
#     input_shape = x.shape
#
#     try:
#         # Encode input - try with original data first
#         encoded_x = jax.vmap(model.linear_encoder)(x)
#     except (TypeError, ValueError) as e:
#         # If error, print debug info and try reshaping
#         print(f"Input shape: {input_shape}")
#         print(f"Linear encoder weight shape: {model.linear_encoder.weight.shape}")
#
#         # Common reshape approaches:
#         # 1. If x is (batch, seq_len, features) but encoder expects (batch, features)
#         if len(input_shape) == 3:
#             # Try using the first timestep or averaging across time
#             x_reshaped = x[:, 0, :]  # Use first timestep
#             # Alternatively: x_reshaped = jnp.mean(x, axis=1)  # Average across time
#             encoded_x = jax.vmap(model.linear_encoder)(x_reshaped)
#         else:
#             # If that doesn't work, we need more specific handling
#             raise ValueError(f"Unable to match input shape {input_shape} to encoder expectations. Original error: {e}")
#
#     # Rest of the function remains the same...
#     # List to store activations at each time step
#     activations = [encoded_x]
#     current_x = encoded_x
#     current_state = state
#
#     # Run through blocks collecting intermediate activations
#     for i, (block, key) in enumerate(zip(model.blocks, dropkeys)):
#         current_x, current_state = block(current_x, current_state, key=key)
#         current_x = jnp.clip(current_x, -100.0, 100.0)
#         activations.append(current_x)
#
#     # For each layer's activations, compute prediction loss
#     prediction_losses = []
#
#     for t in range(len(activations[0]) - prediction_horizon):
#         # For each layer
#         layer_losses = []
#         for layer_idx, layer_activations in enumerate(activations):
#             # Current activations
#             current = layer_activations[t]
#
#             # Target (future) activations
#             target = layer_activations[t + prediction_horizon]
#
#             # Simple MSE prediction loss
#             layer_loss = jnp.mean((current - target) ** 2)
#             layer_losses.append(layer_loss)
#
#         # Average across layers for this time step
#         prediction_losses.append(jnp.mean(jnp.array(layer_losses)))
#
#     # Average across time steps
#     total_prediction_loss = jnp.mean(jnp.array(prediction_losses))
#
#     # Additional metrics for monitoring
#     metrics = {
#         'prediction_loss': total_prediction_loss,
#     }
#
#     return total_prediction_loss, metrics

# @eqx.filter_jit
# @eqx.filter_value_and_grad(has_aux=True)
# def classification_loss(diff_model, static_model, X, y, state, key):
#     model = eqx.combine(diff_model, static_model)
#
#     # Split key for classification and predictive losses
#     key_class, key_pred = jr.split(key)
#
#     # Calculate standard classification loss
#     pred_y, new_state = calc_output(
#         model, X, state, key_class, model.stateful, model.nondeterministic
#     )
#     class_loss = jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1))
#
#     # Add regularization if applicable
#     norm = 0
#     if hasattr(model, 'lip2') and model.lip2:
#         for layer in model.vf.mlp.layers:
#             norm += jnp.mean(
#                 jnp.linalg.norm(layer.weight, axis=-1)
#                 + jnp.linalg.norm(layer.bias, axis=-1)
#             )
#         norm *= model.lambd
#
#     # Calculate predictive coding loss
#     pred_loss, _ = simple_predictive_loss(model, X, state, key_pred)
#
#     # Combine losses with weighting
#     pred_weight = 0.1  # Start with a small weight
#     total_loss = class_loss + norm + pred_weight * pred_loss
#
#     return total_loss, new_state

# @eqx.filter_jit
# @eqx.filter_value_and_grad(has_aux=True)
# def classification_loss(diff_model, static_model, X, y, state, key):
#     model = eqx.combine(diff_model, static_model)
#
#     # Split key for different random operations
#     key_output, key_predictive = jr.split(key)
#
#     # Calculate standard classification output and loss
#     pred_y, new_state = calc_output(
#         model, X, state, key_output, model.stateful, model.nondeterministic
#     )
#     classification_loss_value = jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1))
#
#     # L2 regularization if applicable
#     norm = 0
#     if hasattr(model, 'lip2') and model.lip2:
#         for layer in model.vf.mlp.layers:
#             norm += jnp.mean(
#                 jnp.linalg.norm(layer.weight, axis=-1)
#                 + jnp.linalg.norm(layer.bias, axis=-1)
#             )
#         norm *= model.lambd
#
#     # Calculate predictive coding loss
#     pred_loss, pred_metrics = predictive_coding_loss(model, X, state, key_predictive)
#
#     # Combine losses with weighting factor for predictive coding
#     # Start with a small weight like 0.1 and adjust based on results
#     predictive_weight = 0.1
#     total_loss = classification_loss_value + norm + predictive_weight * pred_loss
#
#     # Return combined loss and updated state along with metrics for monitoring
#     combined_metrics = {
#         'classification_loss': classification_loss_value,
#         'regularization': norm,
#         'predictive_coding_loss': pred_loss,
#         'total_loss': total_loss,
#         **pred_metrics  # Include any additional metrics from predictive coding
#     }
#
#     return total_loss, (new_state, combined_metrics)

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(diff_model, static_model, X, y, state, key):
    model = eqx.combine(diff_model, static_model)
    pred_y, state = calc_output(
        model, X, state, key, model.stateful, model.nondeterministic
    )
    pred_y = pred_y[:, :, 0]
    norm = 0
    if hasattr(model, 'lip2') and model.lip2:
        for layer in model.vf.mlp.layers:
            norm += jnp.mean(
                jnp.linalg.norm(layer.weight, axis=-1)
                + jnp.linalg.norm(layer.bias, axis=-1)
            )
        norm *= model.lambd
    return (
        jnp.mean(jnp.mean((pred_y - y) ** 2, axis=1)) + norm,
        state,
    )


# 创建一个简化版的BFGS优化器
def create_simple_bfgs(learning_rate=0.01, history_size=10):
    def init_fn(params):
        # Instead of storing the full Hessian, store recent updates
        return {
            's_history': [],  # parameter differences
            'y_history': [],  # gradient differences
            'rho_history': [],  # curvature information
            'prev_params': jax.tree_map(jnp.copy, params),
            'prev_grad': jax.tree_map(jnp.zeros_like, params),
            'step': 0,
            'history_size': history_size
        }

    def update_fn(grads, state, params=None):
        # 将梯度和参数展平
        flat_grads = jnp.concatenate([g.flatten() for g in jax.tree_leaves(grads)])
        flat_params = jnp.concatenate([p.flatten() for p in jax.tree_leaves(params)])

        # 使用当前的逆Hessian近似计算搜索方向
        search_direction = -jnp.matmul(state['inv_hessian'], flat_grads)

        # 计算步长（这里简化为固定学习率）
        step_size = learning_rate

        # 计算参数更新
        param_update = step_size * search_direction

        # 更新逆Hessian近似
        s = flat_params - state['prev_params']  # 参数变化
        y = flat_grads - state['prev_grad']  # 梯度变化

        # 仅在非首次迭代且s和y满足曲率条件时更新
        do_update = (state['step'] > 0) & (jnp.dot(s, y) > 1e-10)

        # BFGS更新公式
        rho = 1.0 / jnp.maximum(jnp.dot(y, s), 1e-10)
        I = jnp.eye(len(flat_grads))

        # 条件更新逆Hessian
        new_inv_hessian = jax.lax.cond(
            do_update,
            lambda _: (I - jnp.outer(s, y) * rho) @ state['inv_hessian'] @ (
                        I - jnp.outer(y, s) * rho) + rho * jnp.outer(s, s),
            lambda _: state['inv_hessian'],
            operand=None
        )

        # 将更新重构回参数树结构
        updates = []
        idx = 0
        for p in jax.tree_leaves(params):
            size = p.size
            update = param_update[idx:idx + size].reshape(p.shape)
            updates.append(update)
            idx += size

        updates = jax.tree_unflatten(jax.tree_structure(params), updates)

        # 更新状态
        new_state = {
            'inv_hessian': new_inv_hessian,
            'prev_params': flat_params,
            'prev_grad': flat_grads,
            'step': state['step'] + 1
        }

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# 使用我们的BFGS优化器的make_step函数
@eqx.filter_jit
def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key):
    diff_model, static_model = eqx.partition(model, filter_spec)
    (value, new_state), grads = loss_fn(diff_model, static_model, X, y, state, key)

    # 获取当前参数
    current_params = eqx.filter(model, filter_spec)

    # 使用BFGS更新
    updates, new_opt_state = opt.update(grads, opt_state, params=current_params)
    updated_model = eqx.apply_updates(model, updates)

    return updated_model, new_state, new_opt_state, value

# @eqx.filter_jit
# def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key):
#     diff_model, static_model = eqx.partition(model, filter_spec)
#     (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key)
#     updates, opt_state = opt.update(grads, opt_state)
#     model = eqx.apply_updates(model, updates)
#     return model, state, opt_state, value


# ## used to get the gradient of c, k and kp
# def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key):
#     diff_model, static_model = eqx.partition(model, filter_spec)
#     (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key)
#     updates, opt_state = opt.update(grads, opt_state)
#
#     # Access the FDTD2DLayer within the model
#     fdtd_layer = model.blocks[0].fdtd  # Assuming the FDTD2DLayer is in the first block
#
#     # Print gradients of c, kp_diag, and k_diag
#     print("Gradients of c:", grads.blocks[0].fdtd.c)
#     print("Gradients of kp_diag:", grads.blocks[0].fdtd.kp_diag)
#     print("Gradients of k_diag:", grads.blocks[0].fdtd.k_diag)

    # # Print parameter values before update
    # print("Parameter values before update:")
    # c_prev = fdtd_layer.c
    # kp_diag_prev = fdtd_layer.kp_diag
    # k_diag_prev = fdtd_layer.k_diag
    # print("c:", fdtd_layer.c)
    # print("kp_diag:", fdtd_layer.kp_diag)
    # print("k_diag:", fdtd_layer.k_diag)
    #
    # model = eqx.apply_updates(model, updates)
    #
    # # Print parameter values after update
    # print("Parameter values after update:")
    # c_new = fdtd_layer.c
    # kp_diag_new = fdtd_layer.kp_diag
    # k_diag_new = fdtd_layer.k_diag
    # print("c:", fdtd_layer.c)
    # print("kp_diag:", fdtd_layer.kp_diag)
    # print("k_diag:", fdtd_layer.k_diag)

    # print('c_changed', c_prev == c_new)
    # print('kp_diag_changed', kp_diag_prev == kp_diag_new)
    # print('k_diag_changed', k_diag_prev == k_diag_new)

    # return model, state, opt_state, value

# def make_step(model, filter_spec, X, y, loss_fn, state, opt, opt_state, key):
#     diff_model, static_model = eqx.partition(model, filter_spec)
#     (value, state), grads = loss_fn(diff_model, static_model, X, y, state, key)
#     updates, opt_state = opt.update(grads, opt_state)
#
#     # Access the FDTD2DLayer within the model
#     fdtd_layer = model.blocks[0].fdtd  # Assuming the FDTD2DLayer is in the first block
#
#     # Save initial parameter values
#     c_prev = fdtd_layer.c
#     kp_diag_prev = fdtd_layer.kp_diag
#     k_diag_prev = fdtd_layer.k_diag
#
#     # Apply updates
#     model = eqx.apply_updates(model, updates)
#
#     # Save new parameter values
#     c_new = fdtd_layer.c
#     kp_diag_new = fdtd_layer.kp_diag
#     k_diag_new = fdtd_layer.k_diag
#
#     # Compare initial and new values
#     c_changed = not jnp.array_equal(c_prev, c_new)
#     kp_diag_changed = not jnp.array_equal(kp_diag_prev, kp_diag_new)
#     k_diag_changed = not jnp.array_equal(k_diag_prev, k_diag_new)
#
#     print('c_changed:', c_changed)
#     print('kp_diag_changed:', kp_diag_changed)
#     print('k_diag_changed:', k_diag_changed)
#
#     return model, state, opt_state, value

'''The original version'''
# def train_model(
#         dataset_name,
#         model,
#         metric,
#         filter_spec,
#         state,
#         dataloaders,
#         num_steps,
#         print_steps,
#         lr,
#         lr_scheduler,
#         batch_size,
#         key,
#         output_dir,
#         id,
# ):
#     if metric == "accuracy":
#         best_val = max
#         operator_improv = lambda x, y: x >= y
#         operator_no_improv = lambda x, y: x <= y
#     elif metric == "mse":
#         best_val = min
#         operator_improv = lambda x, y: x <= y
#         operator_no_improv = lambda x, y: x >= y
#     else:
#         raise ValueError(f"Unknown metric: {metric}")
#
#     if os.path.isdir(output_dir):
#         user_input = input(
#             f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): "
#         )
#         if user_input.lower() == "yes":
#             shutil.rmtree(output_dir)
#             os.makedirs(output_dir)
#             print(f"Directory {output_dir} has been deleted and recreated.")
#         else:
#             raise ValueError(f"Directory {output_dir} already exists. Exiting.")
#     else:
#         os.makedirs(output_dir)
#         print(f"Directory {output_dir} has been created.")
#
#     batchkey, key = jr.split(key, 2)
#     opt = optax.adam(learning_rate=lr_scheduler(lr))
#     # opt = optax.adam(learning_rate=lr)
#     opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
#
#     if model.classification:
#         loss_fn = classification_loss
#     else:
#         loss_fn = regression_loss
#
#     running_loss = 0.0
#     if metric == "accuracy":
#         all_val_metric = [0.0]
#         all_train_metric = [0.0]
#         val_metric_for_best_model = [0.0]
#     elif metric == "mse":
#         all_val_metric = [100.0]
#         all_train_metric = [100.0]
#         val_metric_for_best_model = [100.0]
#     no_val_improvement = 0
#     all_time = []
#     start = time.time()
#     for step, data in zip(
#             range(num_steps),
#             dataloaders["train"].loop(batch_size, key=batchkey),
#     ):
#         stepkey, key = jr.split(key, 2)
#         X, y = data
#         model, state, opt_state, value = make_step(
#             model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey
#         )
#         running_loss += value
#         if (step + 1) % print_steps == 0:
#             predictions = []
#             labels = []
#             for data in dataloaders["train"].loop_epoch(batch_size):
#                 stepkey, key = jr.split(key, 2)
#                 inference_model = eqx.tree_inference(model, value=True)
#                 X, y = data
#                 prediction, _ = calc_output(
#                     inference_model,
#                     X,
#                     state,
#                     stepkey,
#                     model.stateful,
#                     model.nondeterministic,
#                 )
#                 predictions.append(prediction)
#                 labels.append(y)
#             prediction = jnp.vstack(predictions)
#             y = jnp.vstack(labels)
#             if model.classification:
#                 train_metric = jnp.mean(
#                     jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                 )
#             else:
#                 prediction = prediction[:, :, 0]
#                 train_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
#             predictions = []
#             labels = []
#             for data in dataloaders["val"].loop_epoch(batch_size):
#                 stepkey, key = jr.split(key, 2)
#                 inference_model = eqx.tree_inference(model, value=True)
#                 X, y = data
#                 prediction, _ = calc_output(
#                     inference_model,
#                     X,
#                     state,
#                     stepkey,
#                     model.stateful,
#                     model.nondeterministic,
#                 )
#                 predictions.append(prediction)
#                 labels.append(y)
#             prediction = jnp.vstack(predictions)
#             y = jnp.vstack(labels)
#             if model.classification:
#                 val_metric = jnp.mean(
#                     jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                 )
#             else:
#                 prediction = prediction[:, :, 0]
#                 val_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
#             end = time.time()
#             total_time = end - start
#             print(
#                 f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
#                 f"Train metric: {train_metric}, "
#                 f"Validation metric: {val_metric}, Time: {total_time}"
#             )
#             start = time.time()
#             if step > 0:
#                 if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
#                     no_val_improvement += 1
#                     if no_val_improvement > 10:
#                         break
#                 else:
#                     no_val_improvement = 0
#                 if operator_improv(val_metric, best_val(val_metric_for_best_model)):
#                     val_metric_for_best_model.append(val_metric)
#                     predictions = []
#                     labels = []
#                     for data in dataloaders["test"].loop_epoch(batch_size):
#                         stepkey, key = jr.split(key, 2)
#                         inference_model = eqx.tree_inference(model, value=True)
#                         X, y = data
#                         prediction, _ = calc_output(
#                             inference_model,
#                             X,
#                             state,
#                             stepkey,
#                             model.stateful,
#                             model.nondeterministic,
#                         )
#                         predictions.append(prediction)
#                         labels.append(y)
#                     prediction = jnp.vstack(predictions)
#                     y = jnp.vstack(labels)
#                     if model.classification:
#                         test_metric = jnp.mean(
#                             jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                         )
#                     else:
#                         prediction = prediction[:, :, 0]
#                         test_metric = jnp.mean(
#                             jnp.mean((prediction - y) ** 2, axis=1), axis=0
#                         )
#                     print(f"Test metric: {test_metric}")
#                 running_loss = 0.0
#                 all_train_metric.append(train_metric)
#                 all_val_metric.append(val_metric)
#                 all_time.append(total_time)
#                 steps = jnp.arange(0, step + 1, print_steps)
#                 all_train_metric_save = jnp.array(all_train_metric)
#                 all_val_metric_save = jnp.array(all_val_metric)
#                 all_time_save = jnp.array(all_time)
#                 test_metric_save = jnp.array(test_metric)
#                 jnp.save(output_dir + "/steps.npy", steps)
#                 jnp.save(output_dir + "/all_train_metric.npy", all_train_metric_save)
#                 jnp.save(output_dir + "/all_val_metric.npy", all_val_metric_save)
#                 jnp.save(output_dir + "/all_time.npy", all_time_save)
#                 jnp.save(output_dir + "/test_metric.npy", test_metric_save)
#
#     print(f"Test metric: {test_metric}")
#
#     steps = jnp.arange(0, num_steps + 1, print_steps)
#     all_train_metric = jnp.array(all_train_metric)
#     all_val_metric = jnp.array(all_val_metric)
#     all_time = jnp.array(all_time)
#     test_metric = jnp.array(test_metric)
#     jnp.save(output_dir + "/steps.npy", steps)
#     jnp.save(output_dir + "/all_train_metric.npy", all_train_metric)
#     jnp.save(output_dir + "/all_val_metric.npy", all_val_metric)
#     jnp.save(output_dir + "/all_time.npy", all_time)
#     jnp.save(output_dir + "/test_metric.npy", test_metric)
#
#     return model

'''The version used to get the gradient and values of c, kp, and k'''
# def train_model(
#     dataset_name,
#     model,
#     metric,
#     filter_spec,
#     state,
#     dataloaders,
#     num_steps,
#     print_steps,
#     lr,
#     lr_scheduler,
#     batch_size,
#     key,
#     output_dir,
#     id,
# ):
#     if metric == "accuracy":
#         best_val = max
#         operator_improv = lambda x, y: x >= y
#         operator_no_improv = lambda x, y: x <= y
#     elif metric == "mse":
#         best_val = min
#         operator_improv = lambda x, y: x <= y
#         operator_no_improv = lambda x, y: x >= y
#     else:
#         raise ValueError(f"Unknown metric: {metric}")
#
#     if os.path.isdir(output_dir):
#         user_input = input(
#             f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): "
#         )
#         if user_input.lower() == "yes":
#             shutil.rmtree(output_dir)
#             os.makedirs(output_dir)
#             print(f"Directory {output_dir} has been deleted and recreated.")
#         else:
#             raise ValueError(f"Directory {output_dir} already exists. Exiting.")
#     else:
#         os.makedirs(output_dir)
#         print(f"Directory {output_dir} has been created.")
#
#     batchkey, key = jr.split(key, 2)
#     opt = optax.adam(learning_rate=lr_scheduler(lr))
#     opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
#
#     if model.classification:
#         loss_fn = classification_loss
#     else:
#         loss_fn = regression_loss
#
#     running_loss = 0.0
#     if metric == "accuracy":
#         all_val_metric = [0.0]
#         all_train_metric = [0.0]
#         val_metric_for_best_model = [0.0]
#     elif metric == "mse":
#         all_val_metric = [100.0]
#         all_train_metric = [100.0]
#         val_metric_for_best_model = [100.0]
#     no_val_improvement = 0
#     all_time = []
#     start = time.time()
#
#     for step, data in zip(
#         range(num_steps),
#         dataloaders["train"].loop(batch_size, key=batchkey),
#     ):
#         stepkey, key = jr.split(key, 2)
#         X, y = data
#         model, state, opt_state, value = make_step(
#             model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey
#         )
#         running_loss += value
#
#         if (step + 1) % print_steps == 0:
#             predictions = []
#             labels = []
#             for data in dataloaders["train"].loop_epoch(batch_size):
#                 stepkey, key = jr.split(key, 2)
#                 inference_model = eqx.tree_inference(model, value=True)
#                 X, y = data
#                 prediction, _ = calc_output(
#                     inference_model,
#                     X,
#                     state,
#                     stepkey,
#                     model.stateful,
#                     model.nondeterministic,
#                 )
#                 predictions.append(prediction)
#                 labels.append(y)
#             prediction = jnp.vstack(predictions)
#             y = jnp.vstack(labels)
#             if model.classification:
#                 train_metric = jnp.mean(
#                     jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                 )
#             else:
#                 prediction = prediction[:, :, 0]
#                 train_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
#
#             predictions = []
#             labels = []
#             for data in dataloaders["val"].loop_epoch(batch_size):
#                 stepkey, key = jr.split(key, 2)
#                 inference_model = eqx.tree_inference(model, value=True)
#                 X, y = data
#                 prediction, _ = calc_output(
#                     inference_model,
#                     X,
#                     state,
#                     stepkey,
#                     model.stateful,
#                     model.nondeterministic,
#                 )
#                 predictions.append(prediction)
#                 labels.append(y)
#             prediction = jnp.vstack(predictions)
#             y = jnp.vstack(labels)
#             if model.classification:
#                 val_metric = jnp.mean(
#                     jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                 )
#             else:
#                 prediction = prediction[:, :, 0]
#                 val_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
#
#             end = time.time()
#             total_time = end - start
#             print(
#                 f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
#                 f"Train metric: {train_metric}, "
#                 f"Validation metric: {val_metric}, Time: {total_time}"
#             )
#             start = time.time()
#
#             if step > 0:
#                 if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
#                     no_val_improvement += 1
#                     if no_val_improvement > 10:
#                         break
#                 else:
#                     no_val_improvement = 0
#                 if operator_improv(val_metric, best_val(val_metric_for_best_model)):
#                     val_metric_for_best_model.append(val_metric)
#
#             predictions = []
#             labels = []
#             for data in dataloaders["test"].loop_epoch(batch_size):
#                 stepkey, key = jr.split(key, 2)
#                 inference_model = eqx.tree_inference(model, value=True)
#                 X, y = data
#                 prediction, _ = calc_output(
#                     inference_model,
#                     X,
#                     state,
#                     stepkey,
#                     model.stateful,
#                     model.nondeterministic,
#                 )
#                 predictions.append(prediction)
#                 labels.append(y)
#             prediction = jnp.vstack(predictions)
#             y = jnp.vstack(labels)
#             if model.classification:
#                 test_metric = jnp.mean(
#                     jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                 )
#             else:
#                 prediction = prediction[:, :, 0]
#                 test_metric = jnp.mean(
#                     jnp.mean((prediction - y) ** 2, axis=1), axis=0
#                 )
#             print(f"Test metric: {test_metric}")
#
#             running_loss = 0.0
#             all_train_metric.append(train_metric)
#             all_val_metric.append(val_metric)
#             all_time.append(total_time)
#             steps = jnp.arange(0, step + 1, print_steps)
#             all_train_metric_save = jnp.array(all_train_metric)
#             all_val_metric_save = jnp.array(all_val_metric)
#             all_time_save = jnp.array(all_time)
#             test_metric_save = jnp.array(test_metric)
#             jnp.save(output_dir + "/steps.npy", steps)
#             jnp.save(output_dir + "/all_train_metric.npy", all_train_metric_save)
#             jnp.save(output_dir + "/all_val_metric.npy", all_val_metric_save)
#             jnp.save(output_dir + "/all_time.npy", all_time_save)
#             jnp.save(output_dir + "/test_metric.npy", test_metric_save)
#
#     return model


def train_model(
        dataset_name,
        model,
        metric,
        filter_spec,
        state,
        dataloaders,
        num_steps,
        print_steps,
        lr,
        lr_scheduler,
        batch_size,
        key,
        output_dir,
        id,
        dataset=None,  # Add dataset parameter to access the data
):
    if metric == "accuracy":
        best_val = max
        operator_improv = lambda x, y: x >= y
        operator_no_improv = lambda x, y: x <= y
    elif metric == "mse":
        best_val = min
        operator_improv = lambda x, y: x <= y
        operator_no_improv = lambda x, y: x >= y
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if os.path.isdir(output_dir):
        user_input = input(
            f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): "
        )
        if user_input.lower() == "yes":
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            print(f"Directory {output_dir} has been deleted and recreated.")
        else:
            raise ValueError(f"Directory {output_dir} already exists. Exiting.")
    else:
        os.makedirs(output_dir)
        print(f"Directory {output_dir} has been created.")

    batchkey, key = jr.split(key, 2)
    opt = optax.adam(learning_rate=lr_scheduler(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    if model.classification:
        loss_fn = classification_loss
    else:
        loss_fn = regression_loss

    running_loss = 0.0
    if metric == "accuracy":
        all_val_metric = [0.0]
        all_train_metric = [0.0]
        val_metric_for_best_model = [0.0]
    elif metric == "mse":
        all_val_metric = [100.0]
        all_train_metric = [100.0]
        val_metric_for_best_model = [100.0]
    no_val_improvement = 0
    all_time = []
    start = time.time()

    # Save the best model
    best_model = model

    for step, data in zip(
            range(num_steps),
            dataloaders["train"].loop(batch_size, key=batchkey),
    ):
        stepkey, key = jr.split(key, 2)
        X, y = data
        model, state, opt_state, value = make_step(
            model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey
        )
        running_loss += value
        if (step + 1) % print_steps == 0:
            predictions = []
            labels = []
            for data in dataloaders["train"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            if model.classification:
                train_metric = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
            else:
                prediction = prediction[:, :, 0]
                train_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
            predictions = []
            labels = []
            for data in dataloaders["val"].loop_epoch(batch_size):
                stepkey, key = jr.split(key, 2)
                inference_model = eqx.tree_inference(model, value=True)
                X, y = data
                prediction, _ = calc_output(
                    inference_model,
                    X,
                    state,
                    stepkey,
                    model.stateful,
                    model.nondeterministic,
                )
                predictions.append(prediction)
                labels.append(y)
            prediction = jnp.vstack(predictions)
            y = jnp.vstack(labels)
            if model.classification:
                val_metric = jnp.mean(
                    jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                )
            else:
                prediction = prediction[:, :, 0]
                val_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
            end = time.time()
            total_time = end - start
            print(
                f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
                f"Train metric: {train_metric}, "
                f"Validation metric: {val_metric}, Time: {total_time}"
            )
            start = time.time()
            if step > 0:
                if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
                    no_val_improvement += 1
                    if no_val_improvement > 10:
                        break
                else:
                    no_val_improvement = 0
                if operator_improv(val_metric, best_val(val_metric_for_best_model)):
                    val_metric_for_best_model.append(val_metric)
                    predictions = []
                    labels = []
                    for data in dataloaders["test"].loop_epoch(batch_size):
                        stepkey, key = jr.split(key, 2)
                        inference_model = eqx.tree_inference(model, value=True)
                        X, y = data
                        prediction, _ = calc_output(
                            inference_model,
                            X,
                            state,
                            stepkey,
                            model.stateful,
                            model.nondeterministic,
                        )
                        predictions.append(prediction)
                        labels.append(y)
                    prediction = jnp.vstack(predictions)
                    y = jnp.vstack(labels)
                    if model.classification:
                        test_metric = jnp.mean(
                            jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
                        )
                    else:
                        prediction = prediction[:, :, 0]
                        test_metric = jnp.mean(
                            jnp.mean((prediction - y) ** 2, axis=1), axis=0
                        )
                    print(f"Test metric: {test_metric}")

                    # Save the current model as the best model
                    best_model = model

                running_loss = 0.0
                all_train_metric.append(train_metric)
                all_val_metric.append(val_metric)
                all_time.append(total_time)
                steps = jnp.arange(0, step + 1, print_steps)
                all_train_metric_save = jnp.array(all_train_metric)
                all_val_metric_save = jnp.array(all_val_metric)
                all_time_save = jnp.array(all_time)
                test_metric_save = jnp.array(test_metric)
                jnp.save(output_dir + "/steps.npy", steps)
                jnp.save(output_dir + "/all_train_metric.npy", all_train_metric_save)
                jnp.save(output_dir + "/all_val_metric.npy", all_val_metric_save)
                jnp.save(output_dir + "/all_time.npy", all_time_save)
                jnp.save(output_dir + "/test_metric.npy", test_metric_save)

                # Save the checkpoint of the current best model
                eqx.tree_serialise_leaves(output_dir + "/best_model.eqx", best_model)

                # Also save the current state
                eqx.tree_serialise_leaves(output_dir + "/best_state.eqx", state)

    print(f"Test metric: {test_metric}")

    steps = jnp.arange(0, num_steps + 1, print_steps)
    all_train_metric = jnp.array(all_train_metric)
    all_val_metric = jnp.array(all_val_metric)
    all_time = jnp.array(all_time)
    test_metric = jnp.array(test_metric)
    jnp.save(output_dir + "/steps.npy", steps)
    jnp.save(output_dir + "/all_train_metric.npy", all_train_metric)
    jnp.save(output_dir + "/all_val_metric.npy", all_val_metric)
    jnp.save(output_dir + "/all_time.npy", all_time)
    jnp.save(output_dir + "/test_metric.npy", test_metric)

    # Save the final model checkpoint
    eqx.tree_serialise_leaves(output_dir + "/final_model.eqx", model)
    eqx.tree_serialise_leaves(output_dir + "/final_state.eqx", state)

    # Make sure to also save the best model (it might be different from the final model)
    if not os.path.exists(output_dir + "/best_model.eqx"):
        eqx.tree_serialise_leaves(output_dir + "/best_model.eqx", best_model)
        eqx.tree_serialise_leaves(output_dir + "/best_state.eqx", state)

    # # Save the dataset presplit information
    # if dataset is not None:
    #     # Extract seed from the output directory path
    #     # Assume the output_dir has a pattern like "seed_{seed}_lr_{lr}_T_{T:.2f}_params_{param_hash}"
    #     import re
    #     seed_match = re.search(r'seed_(\d+)', os.path.basename(output_dir))
    #     seed = seed_match.group(1) if seed_match else "unknown_seed"
    #
    #     # Create main presplit directory if it doesn't exist yet
    #     main_presplit_dir = os.path.join('/home/zhyuan/Desktop/FTDToss/data_dir', f"processed/UEA/{dataset_name}")
    #     os.makedirs(main_presplit_dir, exist_ok=True)
    #
    #     # Create seed-specific subfolder for presplit data
    #     presplit_dir = os.path.join(main_presplit_dir, f"seed_{seed}")
    #     os.makedirs(presplit_dir, exist_ok=True)
    #
    #     # Save the presplit data
    #     if hasattr(dataset, 'raw_dataloaders'):
    #         # Extract the data from the dataloaders
    #         X_train = dataset.raw_dataloaders['train'].data
    #         y_train = dataset.raw_dataloaders['train'].labels
    #         X_val = dataset.raw_dataloaders['val'].data
    #         y_val = dataset.raw_dataloaders['val'].labels
    #         X_test = dataset.raw_dataloaders['test'].data
    #         y_test = dataset.raw_dataloaders['test'].labels
    #
    #         # Save the data
    #         with open(os.path.join(presplit_dir, "X_train.pkl"), "wb") as f:
    #             pickle.dump(X_train, f)
    #         with open(os.path.join(presplit_dir, "y_train.pkl"), "wb") as f:
    #             pickle.dump(y_train, f)
    #         with open(os.path.join(presplit_dir, "X_val.pkl"), "wb") as f:
    #             pickle.dump(X_val, f)
    #         with open(os.path.join(presplit_dir, "y_val.pkl"), "wb") as f:
    #             pickle.dump(y_val, f)
    #         with open(os.path.join(presplit_dir, "X_test.pkl"), "wb") as f:
    #             pickle.dump(X_test, f)
    #         with open(os.path.join(presplit_dir, "y_test.pkl"), "wb") as f:
    #             pickle.dump(y_test, f)
    #
    #         print(f"Dataset presplit information saved to {presplit_dir}")

    return model

def create_dataset_model_and_train(
        seed,
        data_dir,
        use_presplit,
        dataset_name,
        output_step,
        metric,
        include_time,
        T,
        model_name,
        stepsize,
        logsig_depth,
        linoss_discretization,
        model_args,
        num_steps,
        print_steps,
        lr,
        lr_scheduler,
        batch_size,
        output_parent_dir="",
        id=None,
        save_presplit=True,  # New parameter to control whether to save presplit data
):
    # Determine model directory name with potential suffixes
    if model_name == 'LinOSS':
        model_name_directory = model_name + '_' + linoss_discretization
    elif model_name == 'FDTD':
        # Include FDTD-specific parameters in the directory name if specified
        fdtd_params = ""
        if 'fdtd_wave_speed' in model_args and model_args['fdtd_wave_speed'] is not None:
            fdtd_params += f"_c_{model_args['fdtd_wave_speed']:.2f}"
        if 'fdtd_damping_p' in model_args and model_args['fdtd_damping_p'] is not None:
            fdtd_params += f"_dp_{model_args['fdtd_damping_p']:.4f}"
        if 'fdtd_damping_v' in model_args and model_args['fdtd_damping_v'] is not None:
            fdtd_params += f"_dv_{model_args['fdtd_damping_v']:.4f}"
        model_name_directory = model_name + fdtd_params
    else:
        model_name_directory = model_name

    # Create a shorter output directory path
    output_parent_dir += "FDTD_outputs_Euler/" + model_name_directory + "/" + dataset_name

    # Use a more concise output directory name with just the essential parameters
    output_dir = f"seed_{seed}_lr_{lr}_T_{T:.2f}"

    # Create a unique hash for additional parameters to keep the directory name short
    import hashlib
    param_dict = {
        "include_time": include_time,
        "num_steps": num_steps,
        "model_args": {k: str(v) for k, v in model_args.items()},
        "stepsize": stepsize,
        "logsig_depth": logsig_depth,
    }

    # Add model-specific parameters
    if model_name == "log_ncde" or model_name == "nrde":
        param_dict["stepsize"] = stepsize
        param_dict["logsig_depth"] = logsig_depth

    # Create a hash of the parameters
    import json
    param_str = json.dumps(param_dict, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    output_dir += f"_params_{param_hash}"

    # Full output directory path
    full_output_dir = os.path.join(output_parent_dir, output_dir)

    # Save the full parameter details to a JSON file
    os.makedirs(os.path.dirname(full_output_dir), exist_ok=True)
    param_file = os.path.join(os.path.dirname(full_output_dir), f"params_{param_hash}.json")
    if not os.path.exists(param_file):
        with open(param_file, "w") as f:
            json.dump(param_dict, f, indent=2, default=str)

    key = jr.PRNGKey(seed)

    datasetkey, modelkey, trainkey, key = jr.split(key, 4)
    print(f"Creating dataset {dataset_name}")

    dataset = create_dataset(
        data_dir,
        dataset_name,
        stepsize=stepsize,
        depth=logsig_depth,
        include_time=include_time,
        T=T,
        use_idxs=False,
        use_presplit=use_presplit,
        key=datasetkey,
    )

    print(f"Creating model {model_name}")
    classification = metric == "accuracy"
    filtered_model_args = {
        k: v for k, v in model_args.items() if not k.startswith("fdtd_")
    }
    model, state = create_model(
        model_name,
        dataset.data_dim,
        dataset.logsig_dim,
        logsig_depth,
        dataset.intervals,
        dataset.label_dim,
        classification=classification,
        output_step=output_step,
        linoss_discretization=linoss_discretization,
        **filtered_model_args,
        key=modelkey,
    )
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)

    if model_name == "nrde" or model_name == "log_ncde":
        dataloaders = dataset.path_dataloaders
        if model_name == "log_ncde":
            where = lambda model: (model.intervals, model.pairs)
            filter_spec = eqx.tree_at(
                where, filter_spec, replace=(False, False), is_leaf=lambda x: x is None
            )
        elif model_name == "nrde":
            where = lambda model: (model.intervals,)
            filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))
    elif model_name == "ncde":
        dataloaders = dataset.coeff_dataloaders
    else:
        dataloaders = dataset.raw_dataloaders

    return train_model(
        dataset_name,
        model,
        metric,
        filter_spec,
        state,
        dataloaders,
        num_steps,
        print_steps,
        lr,
        lr_scheduler,
        batch_size,
        trainkey,
        full_output_dir,
        id,
        dataset=dataset if save_presplit else None,  # Only pass dataset if save_presplit is True
    )

"""best one"""
# def train_model(
#         dataset_name,
#         model,
#         metric,
#         filter_spec,
#         state,
#         dataloaders,
#         num_steps,
#         print_steps,
#         lr,
#         lr_scheduler,
#         batch_size,
#         key,
#         output_dir,
#         id,
# ):
#     if metric == "accuracy":
#         best_val = max
#         operator_improv = lambda x, y: x >= y
#         operator_no_improv = lambda x, y: x <= y
#     elif metric == "mse":
#         best_val = min
#         operator_improv = lambda x, y: x <= y
#         operator_no_improv = lambda x, y: x >= y
#     else:
#         raise ValueError(f"Unknown metric: {metric}")
#
#     if os.path.isdir(output_dir):
#         user_input = input(
#             f"Warning: Output directory {output_dir} already exists. Do you want to delete it? (yes/no): "
#         )
#         if user_input.lower() == "yes":
#             shutil.rmtree(output_dir)
#             os.makedirs(output_dir)
#             print(f"Directory {output_dir} has been deleted and recreated.")
#         else:
#             raise ValueError(f"Directory {output_dir} already exists. Exiting.")
#     else:
#         os.makedirs(output_dir)
#         print(f"Directory {output_dir} has been created.")
#
#     batchkey, key = jr.split(key, 2)
#     opt = optax.adam(learning_rate=lr_scheduler(lr))
#     # opt = optax.adam(learning_rate=lr)
#     opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
#
#     # opt = create_simple_bfgs(learning_rate=lr_scheduler(lr))
#     # opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
#
#     if model.classification:
#         loss_fn = classification_loss
#     else:
#         loss_fn = regression_loss
#
#     running_loss = 0.0
#     if metric == "accuracy":
#         all_val_metric = [0.0]
#         all_train_metric = [0.0]
#         val_metric_for_best_model = [0.0]
#     elif metric == "mse":
#         all_val_metric = [100.0]
#         all_train_metric = [100.0]
#         val_metric_for_best_model = [100.0]
#     no_val_improvement = 0
#     all_time = []
#     start = time.time()
#
#     # Save the best model
#     best_model = model
#
#     for step, data in zip(
#             range(num_steps),
#             dataloaders["train"].loop(batch_size, key=batchkey),
#     ):
#         stepkey, key = jr.split(key, 2)
#         X, y = data
#         model, state, opt_state, value = make_step(
#             model, filter_spec, X, y, loss_fn, state, opt, opt_state, stepkey
#         )
#         running_loss += value
#         if (step + 1) % print_steps == 0:
#             predictions = []
#             labels = []
#             for data in dataloaders["train"].loop_epoch(batch_size):
#                 stepkey, key = jr.split(key, 2)
#                 inference_model = eqx.tree_inference(model, value=True)
#                 X, y = data
#                 prediction, _ = calc_output(
#                     inference_model,
#                     X,
#                     state,
#                     stepkey,
#                     model.stateful,
#                     model.nondeterministic,
#                 )
#                 predictions.append(prediction)
#                 labels.append(y)
#             prediction = jnp.vstack(predictions)
#             y = jnp.vstack(labels)
#             if model.classification:
#                 train_metric = jnp.mean(
#                     jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                 )
#             else:
#                 prediction = prediction[:, :, 0]
#                 train_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
#             predictions = []
#             labels = []
#             for data in dataloaders["val"].loop_epoch(batch_size):
#                 stepkey, key = jr.split(key, 2)
#                 inference_model = eqx.tree_inference(model, value=True)
#                 X, y = data
#                 prediction, _ = calc_output(
#                     inference_model,
#                     X,
#                     state,
#                     stepkey,
#                     model.stateful,
#                     model.nondeterministic,
#                 )
#                 predictions.append(prediction)
#                 labels.append(y)
#             prediction = jnp.vstack(predictions)
#             y = jnp.vstack(labels)
#             if model.classification:
#                 val_metric = jnp.mean(
#                     jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                 )
#             else:
#                 prediction = prediction[:, :, 0]
#                 val_metric = jnp.mean(jnp.mean((prediction - y) ** 2, axis=1), axis=0)
#             end = time.time()
#             total_time = end - start
#             print(
#                 f"Step: {step + 1}, Loss: {running_loss / print_steps}, "
#                 f"Train metric: {train_metric}, "
#                 f"Validation metric: {val_metric}, Time: {total_time}"
#             )
#             start = time.time()
#             if step > 0:
#                 if operator_no_improv(val_metric, best_val(val_metric_for_best_model)):
#                     no_val_improvement += 1
#                     if no_val_improvement > 10:
#                         break
#                 else:
#                     no_val_improvement = 0
#                 if operator_improv(val_metric, best_val(val_metric_for_best_model)):
#                     val_metric_for_best_model.append(val_metric)
#                     predictions = []
#                     labels = []
#                     for data in dataloaders["test"].loop_epoch(batch_size):
#                         stepkey, key = jr.split(key, 2)
#                         inference_model = eqx.tree_inference(model, value=True)
#                         X, y = data
#                         prediction, _ = calc_output(
#                             inference_model,
#                             X,
#                             state,
#                             stepkey,
#                             model.stateful,
#                             model.nondeterministic,
#                         )
#                         predictions.append(prediction)
#                         labels.append(y)
#                     prediction = jnp.vstack(predictions)
#                     y = jnp.vstack(labels)
#                     if model.classification:
#                         test_metric = jnp.mean(
#                             jnp.argmax(prediction, axis=1) == jnp.argmax(y, axis=1)
#                         )
#                     else:
#                         prediction = prediction[:, :, 0]
#                         test_metric = jnp.mean(
#                             jnp.mean((prediction - y) ** 2, axis=1), axis=0
#                         )
#                     print(f"Test metric: {test_metric}")
#
#                     # Save the current model as the best model
#                     best_model = model
#
#                 running_loss = 0.0
#                 all_train_metric.append(train_metric)
#                 all_val_metric.append(val_metric)
#                 all_time.append(total_time)
#                 steps = jnp.arange(0, step + 1, print_steps)
#                 all_train_metric_save = jnp.array(all_train_metric)
#                 all_val_metric_save = jnp.array(all_val_metric)
#                 all_time_save = jnp.array(all_time)
#                 test_metric_save = jnp.array(test_metric)
#                 jnp.save(output_dir + "/steps.npy", steps)
#                 jnp.save(output_dir + "/all_train_metric.npy", all_train_metric_save)
#                 jnp.save(output_dir + "/all_val_metric.npy", all_val_metric_save)
#                 jnp.save(output_dir + "/all_time.npy", all_time_save)
#                 jnp.save(output_dir + "/test_metric.npy", test_metric_save)
#
#                 # Save the checkpoint of the current best model
#                 eqx.tree_serialise_leaves(output_dir + "/best_model.eqx", best_model)
#
#                 # Also save the current state
#                 eqx.tree_serialise_leaves(output_dir + "/best_state.eqx", state)
#
#     print(f"Test metric: {test_metric}")
#
#     steps = jnp.arange(0, num_steps + 1, print_steps)
#     all_train_metric = jnp.array(all_train_metric)
#     all_val_metric = jnp.array(all_val_metric)
#     all_time = jnp.array(all_time)
#     test_metric = jnp.array(test_metric)
#     jnp.save(output_dir + "/steps.npy", steps)
#     jnp.save(output_dir + "/all_train_metric.npy", all_train_metric)
#     jnp.save(output_dir + "/all_val_metric.npy", all_val_metric)
#     jnp.save(output_dir + "/all_time.npy", all_time)
#     jnp.save(output_dir + "/test_metric.npy", test_metric)
#
#     # Save the final model checkpoint
#     eqx.tree_serialise_leaves(output_dir + "/final_model.eqx", model)
#     eqx.tree_serialise_leaves(output_dir + "/final_state.eqx", state)
#
#     # Make sure to also save the best model (it might be different from the final model)
#     if not os.path.exists(output_dir + "/best_model.eqx"):
#         eqx.tree_serialise_leaves(output_dir + "/best_model.eqx", best_model)
#         eqx.tree_serialise_leaves(output_dir + "/best_state.eqx", state)
#
#     return model


def analyze_frequency_band(dataset, sampling_rate=None):
    """
    Analyze the frequency band of time series in a dataset.

    Parameters:
    dataset: Dataset object containing time series data
    sampling_rate: The original sampling rate of the data (if known)

    Returns:
    Dictionary with frequency band information
    """
    import numpy as np
    from scipy import signal

    # Get the raw data from train set
    raw_data = dataset.raw_dataloaders['train'].data

    # If time is included as first dimension, remove it
    if dataset.data_dim > 1 and hasattr(raw_data, 'shape') and len(raw_data.shape) > 2:
        time_series = raw_data[:, :, 1:] if raw_data.shape[-1] > 1 else raw_data
    else:
        time_series = raw_data

    # Calculate time step if sampling_rate not provided
    if sampling_rate is None:
        # Estimate from the intervals
        intervals = dataset.intervals
        time_steps = np.diff(intervals)
        dt = np.mean(time_steps)
        estimated_sampling_rate = 1 / dt
        sampling_rate = estimated_sampling_rate

    # Compute spectral characteristics for each dimension
    results = {}
    results['sampling_rate'] = sampling_rate
    results['nyquist_frequency'] = sampling_rate / 2.0

    # Randomly sample a few time series to analyze
    n_samples = min(10, len(time_series))
    sample_indices = np.random.choice(len(time_series), n_samples, replace=False)

    # Analyze spectral content
    freqs = []
    powers = []

    for idx in sample_indices:
        for dim in range(time_series.shape[-1]):
            signal_data = np.array(time_series[idx, :, dim])
            # Apply FFT
            fft_result = np.fft.rfft(signal_data)
            fft_freq = np.fft.rfftfreq(len(signal_data), d=1 / sampling_rate)

            # Get power spectrum
            power = np.abs(fft_result) ** 2

            # Find dominant frequencies (above 10% of max power)
            threshold = 0.1 * np.max(power)
            significant_mask = power > threshold
            sig_freqs = fft_freq[significant_mask]

            if len(sig_freqs) > 0:
                freqs.extend(sig_freqs)
                powers.extend(power[significant_mask])

    if freqs:
        results['min_frequency'] = min(freqs)
        results['max_frequency'] = max(freqs)
        results['dominant_frequency'] = freqs[np.argmax(powers)]
    else:
        results['min_frequency'] = 0
        results['max_frequency'] = 0
        results['dominant_frequency'] = 0

    return results


# def create_dataset_model_and_train(
#         seed,
#         data_dir,
#         use_presplit,
#         dataset_name,
#         output_step,
#         metric,
#         include_time,
#         T,
#         model_name,
#         stepsize,
#         logsig_depth,
#         linoss_discretization,
#         model_args,
#         num_steps,
#         print_steps,
#         lr,
#         lr_scheduler,
#         batch_size,
#         output_parent_dir="",
#         id=None,
# ):
#     # Determine model directory name with potential suffixes
#     if model_name == 'LinOSS':
#         model_name_directory = model_name + '_' + linoss_discretization
#     elif model_name == 'FDTD':
#         # Include FDTD-specific parameters in the directory name if specified
#         fdtd_params = ""
#         if 'fdtd_wave_speed' in model_args and model_args['fdtd_wave_speed'] is not None:
#             fdtd_params += f"_c_{model_args['fdtd_wave_speed']:.2f}"
#         if 'fdtd_damping_p' in model_args and model_args['fdtd_damping_p'] is not None:
#             fdtd_params += f"_dp_{model_args['fdtd_damping_p']:.4f}"
#         if 'fdtd_damping_v' in model_args and model_args['fdtd_damping_v'] is not None:
#             fdtd_params += f"_dv_{model_args['fdtd_damping_v']:.4f}"
#         model_name_directory = model_name + fdtd_params
#     else:
#         model_name_directory = model_name
#
#     # Create a shorter output directory path
#     output_parent_dir += "FDTD_outputs_ExpA/" + model_name_directory + "/" + dataset_name
#
#     # Use a more concise output directory name with just the essential parameters
#     output_dir = f"seed_{seed}_lr_{lr}_T_{T:.2f}"
#
#     # Create a unique hash for additional parameters to keep the directory name short
#     import hashlib
#     param_dict = {
#         "include_time": include_time,
#         "num_steps": num_steps,
#         "model_args": {k: str(v) for k, v in model_args.items()},
#         "stepsize": stepsize,
#         "logsig_depth": logsig_depth,
#     }
#
#     # Add model-specific parameters
#     if model_name == "log_ncde" or model_name == "nrde":
#         param_dict["stepsize"] = stepsize
#         param_dict["logsig_depth"] = logsig_depth
#
#     # Create a hash of the parameters
#     import json
#     param_str = json.dumps(param_dict, sort_keys=True)
#     param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
#     output_dir += f"_params_{param_hash}"
#
#     # Full output directory path
#     full_output_dir = os.path.join(output_parent_dir, output_dir)
#
#     # Save the full parameter details to a JSON file
#     os.makedirs(os.path.dirname(full_output_dir), exist_ok=True)
#     param_file = os.path.join(os.path.dirname(full_output_dir), f"params_{param_hash}.json")
#     if not os.path.exists(param_file):
#         with open(param_file, "w") as f:
#             json.dump(param_dict, f, indent=2, default=str)
#
#     key = jr.PRNGKey(seed)
#
#     datasetkey, modelkey, trainkey, key = jr.split(key, 4)
#     print(f"Creating dataset {dataset_name}")
#
#     dataset = create_dataset(
#         data_dir,
#         dataset_name,
#         stepsize=stepsize,
#         depth=logsig_depth,
#         include_time=include_time,
#         T=T,
#         use_idxs=False,
#         use_presplit=use_presplit,
#         key=datasetkey,
#     )
#
#     # frequency_info = analyze_frequency_band(dataset)
#     # print(frequency_info)
#
#     print(f"Creating model {model_name}")
#     classification = metric == "accuracy"
#     filtered_model_args = {
#         k: v for k, v in model_args.items() if not k.startswith("fdtd_")
#     }
#     model, state = create_model(
#         model_name,
#         dataset.data_dim,
#         dataset.logsig_dim,
#         logsig_depth,
#         dataset.intervals,
#         dataset.label_dim,
#         classification=classification,
#         output_step=output_step,
#         linoss_discretization=linoss_discretization,
#         **filtered_model_args,
#         key=modelkey,
#     )
#     filter_spec = jax.tree_util.tree_map(lambda _: True, model)
#
#
#     if model_name == "nrde" or model_name == "log_ncde":
#         dataloaders = dataset.path_dataloaders
#         if model_name == "log_ncde":
#             where = lambda model: (model.intervals, model.pairs)
#             filter_spec = eqx.tree_at(
#                 where, filter_spec, replace=(False, False), is_leaf=lambda x: x is None
#             )
#         elif model_name == "nrde":
#             where = lambda model: (model.intervals,)
#             filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))
#     elif model_name == "ncde":
#         dataloaders = dataset.coeff_dataloaders
#     else:
#         dataloaders = dataset.raw_dataloaders
#
#     return train_model(
#         dataset_name,
#         model,
#         metric,
#         filter_spec,
#         state,
#         dataloaders,
#         num_steps,
#         print_steps,
#         lr,
#         lr_scheduler,
#         batch_size,
#         trainkey,
#         full_output_dir,
#         id,
#     )

# def create_dataset_model_and_train(
#         seed,
#         data_dir,
#         use_presplit,
#         dataset_name,
#         output_step,
#         metric,
#         include_time,
#         T,
#         model_name,
#         stepsize,
#         logsig_depth,
#         linoss_discretization,
#         model_args,
#         num_steps,
#         print_steps,
#         lr,
#         lr_scheduler,
#         batch_size,
#         output_parent_dir="",
#         id=None,
# ):
#     # Determine model directory name with potential suffixes
#     if model_name == 'LinOSS':
#         model_name_directory = model_name + '_' + linoss_discretization
#     elif model_name == 'FDTD':
#         # Include FDTD-specific parameters in the directory name if specified
#         fdtd_params = ""
#         if 'fdtd_wave_speed' in model_args and model_args['fdtd_wave_speed'] is not None:
#             fdtd_params += f"_c_{model_args['fdtd_wave_speed']:.2f}"
#         if 'fdtd_damping_p' in model_args and model_args['fdtd_damping_p'] is not None:
#             fdtd_params += f"_kp_{model_args['fdtd_damping_p']:.2f}"
#         if 'fdtd_damping_v' in model_args and model_args['fdtd_damping_v'] is not None:
#             fdtd_params += f"_kv_{model_args['fdtd_damping_v']:.2f}"
#         model_name_directory = model_name + fdtd_params
#     else:
#         model_name_directory = model_name
#
#     output_parent_dir += "outputs/" + model_name_directory + "/" + dataset_name
#     output_dir = f"T_{T:.2f}_time_{include_time}_nsteps_{num_steps}_lr_{lr}"
#
#     if model_name == "log_ncde" or model_name == "nrde":
#         output_dir += f"_stepsize_{stepsize:.2f}_depth_{logsig_depth}"
#
#     for k, v in model_args.items():
#         # Skip FDTD-specific parameters that are already included in the model_name_directory
#         if model_name == 'FDTD' and k.startswith('fdtd_'):
#             continue
#
#         name = str(v)
#         if "(" in name:
#             name = name.split("(", 1)[0]
#         if name == "dt0":
#             output_dir += f"_{k}_" + f"{v:.2f}"
#         else:
#             output_dir += f"_{k}_" + name
#         if name == "PIDController":
#             output_dir += f"_rtol_{v.rtol}_atol_{v.atol}"
#
#     output_dir += f"_seed_{seed}"
#
#     key = jr.PRNGKey(seed)
#
#     datasetkey, modelkey, trainkey, key = jr.split(key, 4)
#     print(f"Creating dataset {dataset_name}")
#
#     dataset = create_dataset(
#         data_dir,
#         dataset_name,
#         stepsize=stepsize,
#         depth=logsig_depth,
#         include_time=include_time,
#         T=T,
#         use_idxs=False,
#         use_presplit=use_presplit,
#         key=datasetkey,
#     )
#
#     print(f"Creating model {model_name}")
#     classification = metric == "accuracy"
#     filtered_model_args = {
#         k: v for k, v in model_args.items() if not k.startswith("fdtd_")
#     }
#     model, state = create_model(
#         model_name,
#         dataset.data_dim,
#         dataset.logsig_dim,
#         logsig_depth,
#         dataset.intervals,
#         dataset.label_dim,
#         classification=classification,
#         output_step=output_step,
#         linoss_discretization=linoss_discretization,
#         **filtered_model_args,
#         key=modelkey,
#     )
#     filter_spec = jax.tree_util.tree_map(lambda _: True, model)
#
#     if model_name == "nrde" or model_name == "log_ncde":
#         dataloaders = dataset.path_dataloaders
#         if model_name == "log_ncde":
#             where = lambda model: (model.intervals, model.pairs)
#             filter_spec = eqx.tree_at(
#                 where, filter_spec, replace=(False, False), is_leaf=lambda x: x is None
#             )
#         elif model_name == "nrde":
#             where = lambda model: (model.intervals,)
#             filter_spec = eqx.tree_at(where, filter_spec, replace=(False,))
#     elif model_name == "ncde":
#         dataloaders = dataset.coeff_dataloaders
#     else:
#         dataloaders = dataset.raw_dataloaders
#
#     return train_model(
#         dataset_name,
#         model,
#         metric,
#         filter_spec,
#         state,
#         dataloaders,
#         num_steps,
#         print_steps,
#         lr,
#         lr_scheduler,
#         batch_size,
#         trainkey,
#         output_parent_dir + "/" + output_dir,
#         id,
#     )