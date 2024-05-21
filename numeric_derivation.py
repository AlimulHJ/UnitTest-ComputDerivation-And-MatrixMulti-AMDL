def derive(f, x, h=0.0001):
    # Calculate the derivative of the function f at point x.
    return (f(x + h) - f(x)) / h