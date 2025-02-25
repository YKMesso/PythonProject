import numpy as np
import matplotlib.pyplot as plt
import math

LEARNING_RATE = 0.45
EPOCHS = 100

def step(x):
    if (x > 0):
        return 1
    else:
        return -1;


INPUTS = np.array([[-1,-1,1],
                    [-1,1,1],
                    [1,-1,1],
                    [1,1,1]])

OUTPUTS = np.array([[-1,1,1,1]]).T

print(INPUTS)
print(OUTPUTS)

np.random.seed(1)

WEIGHTS = 2 * np.random.random((3,1)) - 1
print("Random Weights before training \n", WEIGHTS)

errors=[]

for input_item, desired in zip(INPUTS, OUTPUTS):

   #ADALINE_OUTPUT = (input_item[0] * WEIGHTS[0] + input_item[1] * WEIGHTS[1] + input_item[2] * WEIGHTS[2])
   #Simplified Version
    ADALINE_OUTPUT = np.dot(input_item, WEIGHTS)
    ADALINE_OUTPUT = step(ADALINE_OUTPUT)

    ERROR = desired - ADALINE_OUTPUT

    errors.append(ERROR)

    WEIGHTS[0] = WEIGHTS[0] + LEARNING_RATE * ERROR * input_item[0]
    WEIGHTS[1] = WEIGHTS[1] + LEARNING_RATE * ERROR * input_item[1]
    WEIGHTS[2] = WEIGHTS[2] + LEARNING_RATE * ERROR * input_item[2]

    print("New Weights after training\n", WEIGHTS)

    for input_item, desired in zip(INPUTS, OUTPUTS):
        # Feed this input forward and calculate the ADALINE output
        ADALINE_OUTPUT = ((input_item[0] * WEIGHTS[0]) + (input_item[1] * WEIGHTS[1]) + (input_item[2] * WEIGHTS[2]))
        # Run ADALINE_OUTPUT through the step function
        ADALINE_OUTPUT = step(ADALINE_OUTPUT)
        print("Actual ", ADALINE_OUTPUT, "Desired ", desired)

    # Plot the errors to see how we did during training
    ax = plt.subplot(111)
    ax.plot(errors, c='#aaaaff', label='Training Errors')
    ax.set_xscale("log")
    plt.title("ADALINE Errors (2,-2)")
    plt.legend()
    plt.xlabel('Error')
    plt.ylabel('Value')
    plt.show()