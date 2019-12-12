import copy
import concurrent.futures

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from pyslam.losses import L2Loss


class Options:
    """Class for specifying optimization options."""

    def __init__(self):
        self.max_iters = 100
        """Maximum number of iterations before terminating."""
        self.min_update_norm = 1e-6
        """Minimum update norm before terminating."""
        self.min_cost = 1e-12
        """Minimum cost value before terminating."""
        self.min_cost_decrease = 0.9
        """Minimum cost decrease factor to continue optimization."""

        self.linesearch_alpha = 0.8
        """Factor by which line search step size decreases each iteration."""
        self.linesearch_max_iters = 10
        """Maximum number of line search steps."""
        self.linesearch_min_cost_decrease = 0.9
        """Minimum cost decrease factor to continue the line search."""

        self.allow_nondecreasing_steps = False
        """Enable non-decreasing steps to escape local minima."""
        self.max_nondecreasing_steps = 3
        """Maximum number of non-dereasing steps before terminating."""

        self.num_threads = 1
        """Number of threads to use for residual and jacobian evaluation."""


class Problem:
    """Class for building optimization problems."""

    def __init__(self, options=Options()):
        self.options = options
        """Optimization options."""

        self.param_dict = dict()
        """Dictionary of all parameters with their current values."""

        self.residual_blocks = []
        """List of residual blocks."""
        self.block_param_keys = []
        """List of parameter keys in param_dict that each block depends on."""
        self.block_loss_functions = []
        """List of loss functions applied to each block. Default: L2Loss."""

        self.constant_param_keys = []
        """List of parameter keys in param_dict to be held constant."""

        self._update_partition_dict = {}
        """Autogenerated list of update vector ranges corresponding to each parameter."""
        self._covariance_matrix = None
        """Covariance matrix of final parameter estimates."""
        self._cost_history = []
        """History of cost values at each iteration of solve."""

        if self.options.num_threads > 1:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.options.num_threads)
            """Thread pool for parallel evaluations."""

    def add_residual_block(self, block, param_keys, loss=L2Loss()):
        """Add a cost block to the problem."""
        # param_keys must be a list, but don't force the user to create a
        # 1-element list
        if isinstance(param_keys, str):
            param_keys = [param_keys]

        self.residual_blocks.append(block)
        self.block_param_keys.append(param_keys)
        self.block_loss_functions.append(loss)

    def initialize_params(self, param_dict):
        """Initialize the parameters in the problem."""
        # update does a shallow copy, which is no good for immutable parameters
        self.param_dict.update(copy.deepcopy(param_dict))

    def set_parameters_constant(self, param_keys):
        """Hold a list of parameters constant."""
        # param_keys must be a list, but don't force the user to create a
        # 1-element list
        if isinstance(param_keys, str):
            param_keys = [param_keys]

        for key in param_keys:
            if key not in self.constant_param_keys:
                self.constant_param_keys.append(key)

    def set_parameters_variable(self, param_keys):
        """Allow a list of parameters to vary."""
        # param_keys must be a list, but don't force the user to create a
        # 1-element list
        if isinstance(param_keys, str):
            param_keys = [param_keys]

        for key in param_keys:
            if key in self.constant_param_keys:
                self.constant_param_keys.remove(key)

    def eval_cost(self, param_dict=None):
        """Evaluate the cost function using given parameter values."""
        if param_dict is None:
            param_dict = self.param_dict

        cost = 0.
        for block, keys, loss in zip(self.residual_blocks,
                                     self.block_param_keys,
                                     self.block_loss_functions):
            try:
                params = [param_dict[key] for key in keys]
            except KeyError as e:
                print(
                    "Parameter {} has not been initialized".format(e.args[0]))

            residual = block.evaluate(params)
            cost += np.sum(loss.loss(residual))

        return cost

    def solve(self):
        """Solve the problem using Gauss - Newton."""
        self._update_partition_dict = self._get_update_partition_dict()

        cost = self.eval_cost()
        dx = np.array([100])

        optimization_iters = 0
        nondecreasing_steps_taken = 0
        self._cost_history = [cost]

        done_optimization = False

        while not done_optimization:
            optimization_iters += 1
            prev_cost = cost

            dx, cost = self.solve_one_iter()
            # print("Update vector:\n", str(dx))
            # print("Update norm = %f" % np.linalg.norm(dx))

            # Update cost history
            self._cost_history.append(cost)

            # Update parameters
            for k, r in self._update_partition_dict.items():
                self._perturb_by_key(k, dx[r])

            # Check if done optimizing
            done_optimization = optimization_iters > self.options.max_iters or \
                np.linalg.norm(dx) < self.options.min_update_norm or \
                cost < self.options.min_cost

            if self.options.allow_nondecreasing_steps:
                if nondecreasing_steps_taken == 0:
                    best_params = copy.deepcopy(self.param_dict)

                if cost >= self.options.min_cost_decrease * prev_cost:
                    nondecreasing_steps_taken += 1
                else:
                    nondecreasing_steps_taken = 0

                if nondecreasing_steps_taken \
                        >= self.options.max_nondecreasing_steps:
                    done_optimization = True
                    self.param_dict.update(best_params)
            else:
                done_optimization = done_optimization or \
                    cost >= self.options.min_cost_decrease * prev_cost

        return self.param_dict

    def solve_one_iter(self):
        """Solve one iteration of Gauss-Newton."""
        # precision * dx = information
        precision, information, cost = self._get_precision_information_and_cost()
        dx = splinalg.spsolve(precision, information)

        # Backtrack line search
        if self.options.linesearch_max_iters > 0:
            best_step_size, best_cost = self._do_line_search(dx)
        else:
            best_step_size, best_cost = 1., cost

        return best_step_size * dx, best_cost

    def compute_covariance(self):
        """Compute the covariance matrix after solve has terminated."""
        try:
            # Re-evaluate the precision matrix with the final parameters
            precision, _, _ = self._get_precision_information_and_cost()
            self._covariance_matrix = splinalg.inv(precision.tocsc()).toarray()
        except Exception as e:
            print('Covariance computation failed!\n{}'.format(e))

    def get_covariance_block(self, param0, param1):
        """Get the covariance block corresponding to two parameters."""
        try:
            p0_range = self._update_partition_dict[param0]
            p1_range = self._update_partition_dict[param1]
            return np.squeeze(self._covariance_matrix[
                p0_range.start:p0_range.stop, p1_range.start:p1_range.stop])
        except KeyError as e:
            print(
                'Cannot compute covariance for constant parameter {}'.format(e.args[0]))

        return None

    def summary(self, format='brief'):
        """Return a summary of the optimization.

           format='brief' : Number of iterations, initial/final cost
           format='full'  : Initial/final cost and relative change at each iteration
        """
        if not self._cost_history:
            raise ValueError('solve has not yet been called')

        if format is 'brief':
            entry_format_string = 'Iterations: {:3} | Cost: {:12e} --> {:12e}'
            summary = entry_format_string.format(len(self._cost_history),
                                                 self._cost_history[0],
                                                 self._cost_history[-1])

        elif format is 'full':
            header_string = '{:>5s} | {:>12s} --> {:>12s} | {:>10s}\n'.format(
                'Iter', 'Initial cost', 'Final cost', 'Rel change')
            entry_format_string = '{:5} | {:12e} --> {:12e} | {:+10f}\n'
            summary = [header_string, '-' * len(header_string) + '\n']
            for i, ic, fc in zip(range(len(self._cost_history)),
                                 self._cost_history[:-1],
                                 self._cost_history[1:]):
                summary.append(entry_format_string.format(
                    i, ic, fc, (fc - ic) / ic))

            summary = ''.join(summary)
        else:
            raise ValueError(
                'Invalid summary format \'{}\'.'.format(format) +
                'Valid formats are \'brief\' and \'full\'')

        return summary

    def _get_update_partition_dict(self):
        """Helper function to partition the full update vector."""
        update_partition_dict = {}
        prev_key = ''
        for key, param in self.param_dict.items():
            if key not in self.constant_param_keys:
                if hasattr(param, 'dof'):
                    # Check if parameter specifies a tangent space
                    dof = param.dof
                elif hasattr(param, '__len__'):
                    # Check if parameter is a vector
                    dof = len(param)
                else:
                    # Must be a scalar
                    dof = 1

                if not update_partition_dict:
                    update_partition_dict.update({key: range(dof)})
                else:
                    update_partition_dict.update({key: range(
                        update_partition_dict[prev_key].stop,
                        update_partition_dict[prev_key].stop + dof)})

                prev_key = key

        return update_partition_dict

    def _get_precision_information_and_cost(self):
        """Helper function to build the precision matrix and information vector for the Gauss - Newton update. Also returns the total cost."""
        # The Gauss-Newton step is given by
        # (H.T * W * H) dx = -H.T * W * e
        # or
        # precision * dx = information
        #
        # However, in our case, W is subsumed into H and e by the stiffness parameter
        # so instead we have
        # (H'.T * H') dx = -H'.T * e'
        # where H' = sqrt(W) * H and e' = sqrt(W) * e
        #
        # Note that this is an exactly equivalent formulation, but avoids needing
        # to explicitly construct and multiply the (possibly very large) W
        # matrix.
        HT_blocks = [[None for _ in self.residual_blocks]
                     for _ in self.param_dict]
        e_blocks = [None for _ in self.residual_blocks]
        cost_blocks = [None for _ in self.residual_blocks]

        block_cidx_dict = dict(zip(self.param_dict.keys(),
                                   list(range(len(self.param_dict)))))

        if self.options.num_threads > 1:
            # Evaluate residual and jacobian blocks in parallel
            threads = []
            for block_ridx, (block, keys, loss) in \
                enumerate(zip(self.residual_blocks,
                              self.block_param_keys,
                              self.block_loss_functions)):

                threads.append(self._thread_pool.submit(
                    self._populate_residual_jacobian_and_cost_blocks,
                    HT_blocks, e_blocks, cost_blocks,
                    block_cidx_dict, block_ridx,
                    block, keys, loss))

            concurrent.futures.wait(threads)
        else:
            # Single thread: Call directly instead of submitting a job
            for block_ridx, (block, keys, loss) in \
                enumerate(zip(self.residual_blocks,
                              self.block_param_keys,
                              self.block_loss_functions)):

                self._populate_residual_jacobian_and_cost_blocks(
                    HT_blocks, e_blocks, cost_blocks,
                    block_cidx_dict, block_ridx,
                    block, keys, loss)

        HT = sparse.bmat(HT_blocks, format='csr')
        e = np.squeeze(np.bmat(e_blocks).A)

        precision = HT.dot(HT.T)
        information = -HT.dot(e)
        cost = np.sum(np.array(cost_blocks))

        return precision, information, cost

    def _populate_residual_jacobian_and_cost_blocks(self,
                                                    HT_blocks, e_blocks, cost_blocks,
                                                    block_cidx_dict, block_ridx,
                                                    block, keys, loss):
        params = [self.param_dict[key] for key in keys]
        compute_jacobians = [False if key in self.constant_param_keys
                             else True for key in keys]

        # Drop the residual if all the parameters used to compute it are
        # being held constant
        if any(compute_jacobians):
            residual, jacobians = block.evaluate(params, compute_jacobians)
            # Weight for iteratively reweighted least squares
            sqrt_loss_weight = np.sqrt(loss.weight(residual))

            for key, jac in zip(keys, jacobians):
                if jac is not None:
                    # transposes needed for proper broadcasting
                    HT_blocks[block_cidx_dict[key]][block_ridx] = \
                        sparse.csr_matrix(sqrt_loss_weight.T * jac.T)

            cost_blocks[block_ridx] = np.sum(loss.loss(residual))
            e_blocks[block_ridx] = sqrt_loss_weight * residual

    def _do_line_search(self, dx):
        """Backtrack line search to optimize step size in a given direction."""
        step_size = 1
        best_step_size = step_size
        best_cost = np.inf

        iters = 0
        done_linesearch = False
        while not done_linesearch:
            iters += 1
            test_params = copy.deepcopy(self.param_dict)

            for k, r in self._update_partition_dict.items():
                self._perturb_by_key(k, best_step_size * dx[r], test_params)

            test_cost = self.eval_cost(test_params)

            # print(step_size, " : ", test_cost)

            if iters < self.options.linesearch_max_iters and \
                    test_cost < \
                    self.options.linesearch_min_cost_decrease * best_cost:
                best_cost = test_cost
                best_step_size = step_size
            else:
                if test_cost < best_cost:
                    best_cost = test_cost
                    best_step_size = step_size

                done_linesearch = True

            step_size = self.options.linesearch_alpha * step_size

        # print("Best step size: %f" % best_step_size)
        # print("Best cost: %f" % best_cost)

        return best_step_size, best_cost

    def _perturb_by_key(self, key, dx, param_dict=None):
        """Helper function to update a parameter given an update vector."""
        if param_dict is None:
            param_dict = self.param_dict

        try:
            param_dict[key].perturb(dx)
        except AttributeError:
            # Default vector space behaviour
            param_dict[key] += dx