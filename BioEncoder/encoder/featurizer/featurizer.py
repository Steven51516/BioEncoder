
class Featurizer:
    def __call__(self, x, idx=None):
        """Apply the transformation to x using the specified step or the main transform function."""
        if idx is None:
            return self.transform(x)

        # If there's only one step and idx is 1, then use transform
        if self.get_num_steps() == 1 and idx == 1:
            return self.transform(x)

        # Try to get the specified step function
        step_function = getattr(self, f'step{idx}', None)

        # Check if step function is callable
        if callable(step_function):
            return step_function(x)
        else:
            raise ValueError(f"Step function 'step{idx}' does not exist in this Featurizer.")

    def transform(self, x):
        """Main transformation method."""
        raise NotImplementedError("Transform method needs to be implemented by the child class!")

    def get_num_steps(self):
        """Return the total number of step methods in the featurizer."""
        n = len([name for name in dir(self) if name.startswith('step') and callable(getattr(self, name))])
        return max(n, 1)
