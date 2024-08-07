# PACKAGES
from tensorflow.keras.models import Model
import tensorflow as tf

# LOCAL FUNCTIONS
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import compile_utils
from utils.modelling.custom_losses import MeanSquaredError

# ==============================================================================

class DampingAlgorithm:
    """Default Levenberg–Marquardt damping algorithm.

    This is used inside the LM_Trainer as a generic class. Many damping algorithms
    can be implemented using the same interface.
    """

    def __init__(self,
                 starting_value=1e-3,
                 dec_factor=0.1,
                 inc_factor=10.0,
                 min_value=1e-16,
                 max_value=1e+10,
                 adaptive_scaling=False,
                 fletcher=False):
        """Initializes `DampingAlgorithm` instance.

        Args:
          starting_value: (Optional) Used to initialize the LM_Trainer internal
            damping_factor.
          dec_factor: (Optional) Used in the train_step decrease the
            damping_factor when new_loss < loss.
          inc_factor: (Optional) Used in the train_step increase the
            damping_factor when new_loss >= loss.
          min_value: (Optional) Used as a lower bound for the damping_factor.
            Higher values improve numerical stability in the resolution of the
            linear system, at the cost of slower convergence.
          max_value: (Optional) Used as an upper bound for the damping_factor,
            and as condition to stop the Training process.
          adaptive_scaling: Bool (Optional) Scales the damping_factor adaptively
            multiplying it with max(diagonal(JJ)).
          fletcher: Bool (Optional) Replace the identity matrix with
            diagonal of the gauss-newton hessian approximation, so that there is
            larger movement along the directions where the gradient is smaller.
            This avoids slow convergence in the direction of small gradient.
        """
        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        self.adaptive_scaling = adaptive_scaling
        self.fletcher = fletcher

    def init_step(self, damping_factor, loss):
        return damping_factor

    def decrease(self, damping_factor, loss):
        return tf.math.maximum(
            damping_factor * self.dec_factor,
            self.min_value)

    def increase(self, damping_factor, loss):
        return tf.math.minimum(
            damping_factor * self.inc_factor,
            self.max_value)

    def stop_training(self, damping_factor, loss):
        return damping_factor >= self.max_value

    def apply(self, damping_factor, JJ):
        if self.fletcher:
            damping = tf.linalg.tensor_diag(tf.linalg.diag_part(JJ))
        else:
            damping = tf.eye(tf.shape(JJ)[0], dtype=JJ.dtype)

        scaler = 1.0
        if self.adaptive_scaling:
            scaler = tf.math.reduce_max(tf.linalg.diag_part(JJ))

        damping = tf.scalar_mul(scaler * damping_factor, damping)
        return tf.add(JJ, damping)

class LM_Trainer(Model):
    """Levenberg–Marquardt training algorithm.
    """

    def __init__(self,
                 optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
                 loss=MeanSquaredError(),
                 damping_algorithm=DampingAlgorithm(),
                 attempts_per_step=6,
                 solve_method='solve',
                 jacobian_max_num_rows=100,
                 experimental_use_pfor=True):
        """Initializes `LM_Trainer` instance.

        Args:
          model: It is the Model to be trained, it is expected to inherit
            from tf.keras.Model and to be already built.
          optimizer: (Optional) Performs the update of the model trainable
            variables. When tf.keras.optimizers.SGD is used it is equivalent
            to the operation `w = w - learning_rate * updates`, where updates is
            the step computed using the Levenberg-Marquardt algorithm.
          loss: (Optional) An object which inherits from tf.keras.losses.Loss
          and have an additional function to compute residuals.
          damping_algorithm: (Optional) Class implementing the damping
            algorithm to use during training.
          attempts_per_step: Integer (Optional) During the train step when new
            model variables are computed, the new loss is evaluated and compared
            with the old loss value. If new_loss < loss, then the new variables
            are accepted, otherwise the old variables are restored and
            new ones are computed using a different damping-factor.
            This argument represents the maximum number of attempts, after which
            the step is taken.
          solve_method: (Optional) Possible values are:
            'qr': Uses QR decomposition which is robust but slower.
            'cholesky': Uses Cholesky decomposition which is fast but may fail
                when the hessian approximation is ill-conditioned.
            'solve': Uses tf.linalg.solve. I don't know what algorithm it
                implements. But it seems a compromise in terms of speed and
                robustness.
          jacobian_max_num_rows: Integer (Optional) When the number of residuals
            is greater then the number of variables (overdetermined), the
            hessian approximation is computed by slicing the input and
            accumulate the result of each computation. In this way it is
            possible to drastically reduce the memory usage and increase the
            speed as well. The input is sliced into blocks of size less than or
            equal to the jacobian_max_num_rows.
          experimental_use_pfor: (Optional) If true, vectorizes the jacobian
            computation. Else falls back to a sequential while_loop.
            Vectorization can sometimes fail or lead to excessive memory usage.
            This option can be used to disable vectorization in such cases.
        """
        super(LM_Trainer, self).__init__()

        self.loss = loss
        self.loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        self.optimizer = optimizer
        self.damping_algorithm = damping_algorithm
        self.attempts_per_step = attempts_per_step
        self.jacobian_max_num_rows = jacobian_max_num_rows
        self.experimental_use_pfor = experimental_use_pfor

        # Define and select linear system equation solver.
        def qr(matrix, rhs):
            q, r = tf.linalg.qr(matrix, full_matrices=True)
            y = tf.linalg.matmul(q, rhs, transpose_a=True)
            return tf.linalg.triangular_solve(r, y, lower=False)

        def cholesky(matrix, rhs):
            chol = tf.linalg.cholesky(matrix)
            return tf.linalg.cholesky_solve(chol, rhs)

        def solve(matrix, rhs):
            return tf.linalg.solve(matrix, rhs)

        if solve_method == 'qr':
            self.solve_function = qr
        elif solve_method == 'cholesky':
            self.solve_function = cholesky
        elif solve_method == 'solve':
            self.solve_function = solve
        else:
            raise ValueError('Invalid solve_method.')

        # Keep track of the current damping_factor.
        self.damping_factor = tf.Variable(
            self.damping_algorithm.starting_value,
            trainable=False,
            dtype=self.dtype)


    @tf.function
    def _compute_jacobian(self, inputs, targets):
        with tf.GradientTape(persistent=True) as tape:
            outputs = self(inputs, training=True)
            targets, outputs, _ = compile_utils.match_dtype_and_rank(targets, outputs, None)
            residuals = self.loss.residuals(targets, outputs)

        jacobians = tape.jacobian(
            residuals,
            self.trainable_variables,
            experimental_use_pfor=self.experimental_use_pfor,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        del tape

        num_residuals = tf.size(residuals)
        jacobians = [tf.reshape(j, (num_residuals, -1)) for j in jacobians]
        jacobian = tf.concat(jacobians, axis=1)
        residuals = tf.reshape(residuals, (num_residuals, -1))

        return jacobian, residuals, outputs

    def _init_gauss_newton_overdetermined(self, inputs, targets):
        # Perform the following computation:
        # J, residuals, outputs = self._compute_jacobian(inputs, targets)
        # JJ = tf.linalg.matmul(J, J, transpose_a=True)
        # rhs = tf.linalg.matmul(J, residuals, transpose_a=True)
        #
        # But reduce memory usage by slicing the inputs so that the jacobian
        # matrix will have maximum shape (jacobian_max_num_rows, num_variables)
        # instead of (batch_size, num_variables).
        slice_size = self.jacobian_max_num_rows // self._num_outputs
        if self._num_outputs > self.jacobian_max_num_rows:
            slice_size = 1
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            batch_size = tf.shape(inputs[0])[0]
        else:
            batch_size = tf.shape(inputs)[0]
        num_slices = batch_size // slice_size
        remainder = batch_size % slice_size

        JJ = tf.zeros(
            [self._num_variables, self._num_variables],
            dtype=self.dtype)

        rhs = tf.zeros(
            [self._num_variables, 1],
            dtype=self.dtype)

        outputs_array = tf.TensorArray(
            self.dtype, size=0, dynamic_size=True)

        for i in tf.range(num_slices):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (rhs, tf.TensorShape((self._num_variables, None)))])
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                ninputs = len(inputs)
                _inputs = []
                for l in range(ninputs):
                    _inputs.append(inputs[l][i * slice_size:(i + 1) * slice_size])
            else:
                _inputs = inputs[i * slice_size:(i + 1) * slice_size]

            _targets = targets[i * slice_size:(i + 1) * slice_size]

            J, residuals, _outputs = self._compute_jacobian(_inputs, _targets)

            outputs_array = outputs_array.write(i, _outputs)

            JJ += tf.linalg.matmul(J, J, transpose_a=True)
            rhs += tf.linalg.matmul(J, residuals, transpose_a=True)

        if remainder > 0:
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                ninputs = len(inputs)
                _inputs = []
                for l in range(ninputs):
                    _inputs.append(inputs[l][num_slices * slice_size::])
            else:
                _inputs = inputs[num_slices * slice_size::]

            _targets = targets[num_slices * slice_size::]

            J, residuals, _outputs = self._compute_jacobian(_inputs, _targets)

            if num_slices > 0:
                outputs = tf.concat([outputs_array.concat(), _outputs], axis=0)
            else:
                outputs = _outputs

            JJ += tf.linalg.matmul(J, J, transpose_a=True)
            rhs += tf.linalg.matmul(J, residuals, transpose_a=True)
        else:
            outputs = outputs_array.concat()

        return 0.0, JJ, rhs, outputs

    def _init_gauss_newton_underdetermined(self, inputs, targets):
        J, residuals, outputs = self._compute_jacobian(inputs, targets)
        JJ = tf.linalg.matmul(J, J, transpose_b=True)
        rhs = residuals
        return J, JJ, rhs, outputs

    def _compute_gauss_newton_overdetermined(self, J, JJ, rhs):
        updates = self.solve_function(JJ, rhs)
        return updates

    def _compute_gauss_newton_underdetermined(self, J, JJ, rhs):
        updates = self.solve_function(JJ, rhs)
        updates = tf.linalg.matmul(J, updates, transpose_a=True)
        return updates

    def _train_step(self, inputs, targets,
                    init_gauss_newton, compute_gauss_newton):
        # J: jacobian matrix not used in the overdetermined case.
        # JJ: gauss-newton hessian approximation
        # rhs: gradient when overdetermined, residuals when underdetermined.
        # outputs: prediction of the model for the current inputs.
        J, JJ, rhs, outputs = init_gauss_newton(inputs, targets)

        # Perform normalization for numerical stability.
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            batch_size = tf.shape(inputs[0])[0]
        else:
            batch_size = tf.shape(inputs)[0]
        normalization_factor = 1.0 / tf.dtypes.cast(
            batch_size,
            dtype=self.dtype)

        JJ *= normalization_factor
        rhs *= normalization_factor

        # Compute the current loss value.
        loss = self.loss(targets, outputs)

        stop_training = False
        attempt = 0
        damping_factor = self.damping_algorithm.init_step(
            self.damping_factor, loss)

        attempts = tf.constant(self.attempts_per_step, dtype=tf.int32)

        while tf.constant(True, dtype=tf.bool):
            update_computed = False
            try:
                # Apply the damping to the gauss-newton hessian approximation.
                JJ_damped = self.damping_algorithm.apply(damping_factor, JJ)

                # Compute the updates:
                # overdetermined: updates = (J'*J + damping)^-1*J'*residuals
                # underdetermined: updates = J'*(J*J' + damping)^-1*residuals
                updates = compute_gauss_newton(J, JJ_damped, rhs)
            except Exception as e:
                del e
            else:
                if tf.reduce_all(tf.math.is_finite(updates)):
                    update_computed = True
                    # Split and Reshape the updates
                    updates = tf.split(tf.squeeze(updates, axis=-1), self._splits)
                    updates = [tf.reshape(update, shape)
                               for update, shape in zip(updates, self._shapes)]

                    # Apply the updates to the model trainable_variables.
                    self.optimizer.apply_gradients(
                        zip(updates, self.trainable_variables))

            if attempt < attempts:
                attempt += 1

                if update_computed:
                    # Compute the new loss value.
                    outputs = self(inputs, training=False)
                    new_loss = self.loss(targets, outputs)

                    if new_loss < loss:
                        # Accept the new model variables and backup them.
                        loss = new_loss
                        damping_factor = self.damping_algorithm.decrease(
                            damping_factor, loss)
                        self.backup_variables()
                        break

                    # Restore the old variables and try a new damping_factor.
                    self.restore_variables()

                damping_factor = self.damping_algorithm.increase(
                    damping_factor, loss)

                stop_training = self.damping_algorithm.stop_training(
                    damping_factor, loss)
                if stop_training:
                    break
            else:
                break

        # Update the damping_factor which will be used in the next train_step.
        self.damping_factor.assign(damping_factor)
        return loss, outputs, attempt, stop_training

    def _compute_num_outputs(self, inputs, targets):

        outputs = self(inputs)
        # targets, outputs, _ = compile_utils.match_dtype_and_rank(targets, outputs, None)
        residuals = self.loss.residuals(targets, outputs)
        return tf.reduce_prod(residuals.shape[1::]).numpy()

    def reset_damping_factor(self):
        self.damping_factor.assign(self.damping_algorithm.starting_value)

    def backup_variables(self):
        zip_args = (self.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            backup.assign(variable)

    def restore_variables(self):
        zip_args = (self.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            variable.assign(backup)

    def train_step(self, data):
        inputs, targets = data
        if self._num_outputs is None:
            self._num_outputs = self._compute_num_outputs(inputs, targets)

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            batch_size = tf.shape(inputs[0])[0]
        else:
            batch_size = tf.shape(inputs)[0]
        num_residuals = batch_size * self._num_outputs
        overdetermined = num_residuals >= self._num_variables

        if overdetermined:
            loss, outputs, attempts, stop_training = self._train_step(
                inputs,
                targets,
                self._init_gauss_newton_overdetermined,
                self._compute_gauss_newton_overdetermined)
        else:
            loss, outputs, attempts, stop_training = self._train_step(
                inputs,
                targets,
                self._init_gauss_newton_underdetermined,
                self._compute_gauss_newton_underdetermined)
        self.loss_tracker.update_state(loss)
        return {'train_loss': loss}

