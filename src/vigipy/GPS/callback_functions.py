class Callback:
    """
    A callback class for tracking iterations during optimization.

    This class is used to track and print information at regular intervals during the 
    optimization process. It is designed to be used with optimization routines where 
    a callback function is required.

    N.B.: Currently Callback is not implemented for truncated functions

    Attributes:
        counter (int): A counter that tracks the number of iterations.

    Methods:
        __call__(xk):
            Prints the current iteration number and the value of the variable `xk` 
            every 10 iterations.
    """

    def __init__(self, callback_count=10):
        """
        Initializes the Callback instance with a counter set to 0.
        """
        self.counter = 0
        self.callback_count = int(callback_count) # how often to print output

    def __call__(self, xk):
        """
        Method to be called during each iteration of the optimization process.

        Args:
            xk (numpy.ndarray): The current value of the variable being optimized.
        
        Prints the current iteration number and the value of `xk` every 10 iterations.
        """
        self.counter += 1
        if self.counter % self.callback_count == 0:
            print(f"Iteration {self.counter}: x = {xk}")


class Anneal_Callback:
    """
    A callback class for tracking iterations during simulated annealing.

    This class is used to monitor the progress of simulated annealing algorithms, 
    providing feedback at regular intervals. It is intended for use with optimization 
    routines that require a callback function.

    N.B.: Currently Callback is not implemented for truncated functions

    Attributes:
        counter (int): A counter that tracks the number of iterations.

    Methods:
        __call__(x, f, context):
            Prints the current iteration number, the value of the variable `x`, 
            and the function value `f` every 10 iterations.
    """

    def __init__(self, callback_count=10):
        """
        Initializes the Anneal_Callback instance with a counter set to 0.
        """
        self.counter = 0
        self.callback_count = int(callback_count) # how often to print output

    def __call__(self, x, f, context):
        """
        Method to be called during each iteration of the simulated annealing process.

        Args:
            x (numpy.ndarray): The current value of the variable being optimized.
            f (float): The value of the objective function at `x`.
            context (any): Context or additional information provided by the annealing algorithm.
        
        Prints the current iteration number, the value of `x`, and the function value `f` 
        every 10 iterations.
        """
        self.counter += 1
        if self.counter % self.callback_count == 0:
            print(f"Iteration {self.counter}: x = {x}, f = {f}")

