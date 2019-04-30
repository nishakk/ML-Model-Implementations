from environment import MountainCar
import sys
import numpy

def main(args):
    pass

if __name__ == "__main__":
    main(sys.argv)
    
def updateWeightsParamater(w, state, new_state, action, r, alpha, gamma, bias):
    qsaw = state.dot(w.transpose()[action]) + bias
    act = numpy.zeros(3)
    act[0] = new_state.dot(w.transpose()[0]) + bias
    act[1] = new_state.dot(w.transpose()[1]) + bias
    act[2] = new_state.dot(w.transpose()[2]) + bias
    new_action = numpy.max(act)
    return alpha * (qsaw - (r + (gamma * new_action)))

# Raw mode
def q_learning_raw(mode, episodes, max_iterations, epsilon, gamma, alpha):
    env = MountainCar(mode=mode)

    # Initialize Q-table and bias
    w = numpy.zeros([env.state_space, 3])
    bias = 0
    
    for e in range(episodes):
        state = numpy.zeros([env.state_space])
        state_vals = env.reset()
        state[0] = state_vals[0]
        state[1] = state_vals[1]
        r = 0
        for i in range(max_iterations):
            prob = numpy.random.uniform(0, 1)
            if prob > epsilon:
                    act = numpy.zeros(3)
                    act[0] = state.dot(w.transpose()[0]) + bias
                    act[1] = state.dot(w.transpose()[1]) + bias
                    act[2] = state.dot(w.transpose()[2]) + bias
                    action = numpy.argmax(act)
            else:
                action = numpy.random.choice(3, 1)[0]
            step = env.step(action)
            r = r + step[1]
            new_state = numpy.zeros([env.state_space])
            new_state[0] = step[0][0]
            new_state[1] = step[0][1]
            w_delta = updateWeightsParamater(w, state, new_state, action, step[1], alpha, gamma, bias)
            state = numpy.multiply(state, w_delta)
            w_gradient = numpy.zeros([env.state_space, 3])
            w_gradient[:, action] = state
            w = w - w_gradient
            bias = bias - w_delta
            state = new_state
            if bool(step[2]):
                break
        returns_out.write(str(r) + "\n")
    weight_out.write(str(bias) + "\n")
    for i in range(len(w)):
        for j in range(len(w[0])):
            weight_out.write(str(w[i][j]) + "\n")

# Tile mode
def q_learning_tile(mode, episodes, max_iterations, epsilon, gamma, alpha):
    env = MountainCar(mode=mode)

    # Initialize Q-table and bias
    w = numpy.zeros([env.state_space, 3])
    bias = 0
    for e in range(episodes):
        state = numpy.zeros([env.state_space])
        state_vals = env.reset()
        for key, value in state_vals.iteritems():
            state[key] = 1
        r = 0
        for i in range(max_iterations):
            prob = numpy.random.uniform(0, 1)
            if prob > epsilon:
                    act = numpy.zeros(3)
                    act[0] = state.transpose().dot(w.transpose()[0]) + bias
                    act[1] = state.transpose().dot(w.transpose()[1]) + bias
                    act[2] = state.transpose().dot(w.transpose()[2]) + bias
                    action = numpy.argmax(act)
            else:
                action = numpy.random.choice(3, 1)[0]
            step = env.step(action)
            r = r + step[1]
            new_state = numpy.zeros([env.state_space])
            for key, value in step[0].iteritems():
                new_state[key] = 1
            w_delta = updateWeightsParamater(w, state, new_state, action, step[1], alpha, gamma, bias)
            state = numpy.multiply(state, w_delta)
            w_gradient = numpy.zeros([env.state_space, 3])
            w_gradient[:, action] = state
            w = w - w_gradient
            bias = bias - w_delta
            state = new_state
            if bool(step[2]):
                break
        returns_out.write(str(r) + "\n")
    weight_out.write(str(bias) + "\n")
    for i in range(len(w)):
        for j in range(len(w[0])):
            weight_out.write(str(w[i][j]) + "\n")
    
mode = str(sys.argv[1])
weight_out = open(sys.argv[2], 'w')
returns_out = open(sys.argv[3], 'w')
episodes = int(sys.argv[4])
max_iterations = int(sys.argv[5])
epsilon = float(sys.argv[6])
gamma = float(sys.argv[7])
alpha = float(sys.argv[8])

if mode == "raw":
    q_learning_raw(mode, episodes, max_iterations, epsilon, gamma, alpha)
else:
    q_learning_tile(mode, episodes, max_iterations, epsilon, gamma, alpha)