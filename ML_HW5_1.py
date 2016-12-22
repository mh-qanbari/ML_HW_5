import random
from matplotlib import pyplot as plt
import itertools

g_WORLD_SIZE = (20, 20)
g_RANDOMNESS = 0.15
g_LEARNING_RATE = 0.3
g_DISCOUNT_FACTOR = 0.8
g_MAX_ITERATION = 1700
g_CONVERGE_ITERATION = 1000000

g_ACT_UP = 0
g_ACT_DOWN = 1
g_ACT_RIGHT = 2
g_ACT_LEFT = 3

true = True
false = False


class World:
    def __init__(self, size, blocked_states, goal_state, goal_value):
        # Public:
        width, length = size[:]
        self.size = size
        self.goal_state = goal_state
        self.blocked_states = blocked_states[:]
        # Private:
        self.__QValue = {}
        self.__reward = {}
        value = 1   #  random.random()
        for i in range(width):
            for j in range(length):
                self.__QValue[(i, j, g_ACT_UP)] = value
                self.__QValue[(i, j, g_ACT_DOWN)] = value
                self.__QValue[(i, j, g_ACT_RIGHT)] = value
                self.__QValue[(i, j, g_ACT_LEFT)] = value
        for i in range(width):
            for j in range(length):
                self.__reward[(i, j)] = -1
        self.__reward[(goal_state[0], goal_state[1])] = goal_value
        print "Q(s,a):", self.__QValue
        print "R(s)  :", self.__reward

    def set_value(self, state, action, value):
        self.__QValue[(state[0], state[1], action)] = value

    def get_value(self, state, action):
        return self.__QValue[(state[0], state[1], action)]

    def set_reward(self, state, reward):
        self.__reward[state] = reward

    def get_reward(self, state):
        return self.__reward[state]

    def get_best_action(self, state):
        i, j = state[:]
        best_action = g_ACT_UP
        best_value = self.__QValue[(i, j, g_ACT_UP)]
        down_value = self.__QValue[(i, j, g_ACT_DOWN)]
        right_value = self.__QValue[(i, j, g_ACT_RIGHT)]
        left_value = self.__QValue[(i, j, g_ACT_LEFT)]
        if best_value < down_value:
            best_value = down_value
            best_action = g_ACT_DOWN
        elif best_value < right_value:
            best_value = right_value
            best_action = g_ACT_RIGHT
        elif best_value < left_value:
            best_value = left_value
            best_action = g_ACT_LEFT
        return best_action, best_value

    def print_values(self):
        print "values: ", self.__QValue

    def print_rewards(self):
        print "rewards: ", self.__QValue


class Agent:
    learning_rate = 0
    discount_factor = 0
    converge_iteration = 0
    randomness = 0

    def __init__(self, world, init_state=(0, 0)):
        # Public:
        self.current_state = init_state

        # Private:
        #: :type: World
        self.__world = world

    def go_down(self):
        i, j = self.current_state
        w, l = self.__world.size
        i -= 1
        if (i < 0) or (i >= w) or (j < 0) or (j >= l) or ((i, j) in self.__world.blocked_states):
            return None
        else:
            self.current_state = (i, j)
        return i, j

    def go_left(self):
        i, j = self.current_state
        w, l = self.__world.size
        j -= 1
        if (i < 0) or (i >= w) or (j < 0) or (j >= l) or ((i, j) in self.__world.blocked_states):
            return None
        else:
            self.current_state = (i, j)
        return i, j

    def go_right(self):
        i, j = self.current_state
        w, l = self.__world.size
        j += 1
        if (i < 0) or (i >= w) or (j < 0) or (j >= l) or ((i, j) in self.__world.blocked_states):
            return None
        else:
            self.current_state = (i, j)
        return i, j

    def go_up(self):
        i, j = self.current_state
        w, l = self.__world.size
        i += 1
        if (i < 0) or (i >= w) or (j < 0) or (j >= l) or ((i, j) in self.__world.blocked_states):
            return None
        else:
            self.current_state = (i, j)
        return true

    def take_action(self, action):
        if action == g_ACT_UP:
            return self.go_up()
        if action == g_ACT_DOWN:
            return self.go_down()
        if action == g_ACT_RIGHT:
            return self.go_right()
        if action == g_ACT_LEFT:
            return self.go_left()
        return None

    def start(self):
        history = [self.current_state]
        iter = 0
        while true:
            prev_state = self.current_state
            selected_action = None

            rand_param = random.random()
            if rand_param < Agent.randomness:
                selected_action = random.randint(0, 3)
            else:
                up_value = self.__world.get_value(self.current_state, g_ACT_UP)
                down_value = self.__world.get_value(self.current_state, g_ACT_DOWN)
                right_value = self.__world.get_value(self.current_state, g_ACT_RIGHT)
                left_value = self.__world.get_value(self.current_state, g_ACT_LEFT)
                max_value = up_value
                selected_action = g_ACT_UP
                if down_value > max_value:
                    max_value = down_value
                    selected_action = g_ACT_DOWN
                if right_value > max_value:
                    max_value = right_value
                    selected_action = g_ACT_RIGHT
                if left_value > max_value:
                    max_value = left_value
                    selected_action = g_ACT_LEFT
            act_result = self.take_action(selected_action)

            if act_result is not None:
                old_value = self.__world.get_value(prev_state, selected_action)
                reward = self.__world.get_reward(prev_state)
                best_value = self.__world.get_best_action(self.current_state)[1]
                new_value = old_value + Agent.learning_rate * (reward + Agent.discount_factor * best_value - old_value)
                self.__world.set_value(prev_state, selected_action, new_value)
                history.append(self.current_state)
                # iter += 1
            else:
                old_value = self.__world.get_value(self.current_state, selected_action)
                reward = self.__world.get_reward(self.current_state)
                best_value = self.__world.get_best_action(self.current_state)[1]
                new_value = old_value + Agent.learning_rate * (reward + Agent.discount_factor * best_value - old_value)
                self.__world.set_value(self.current_state, selected_action, new_value)
            iter += 1

            if (iter > Agent.converge_iteration) or (self.current_state == self.__world.goal_state):
                break
        # print history
        return history

if __name__ == "__main__":
    # World initialization
    goal = (0, 19)
    goal_value = 1000
    blocked_states = []
    for i in range(18):
        blocked_states.append((i, 9))
    world = World(g_WORLD_SIZE, blocked_states, goal, goal_value)
    # Agent initialization
    Agent.learning_rate = g_LEARNING_RATE
    Agent.discount_factor = g_DISCOUNT_FACTOR
    Agent.converge_iteration = g_CONVERGE_ITERATION
    Agent.randomness = g_RANDOMNESS
    init_state = (0, 0)
    agent = Agent(world, init_state)
    # Learning block
    episodes_length = []
    for i in range(g_MAX_ITERATION):
        agent.current_state = init_state
        his = agent.start()
        episodes_length.append(len(his))
    N = g_MAX_ITERATION
    menMeans = episodes_length[:]
    menStd = random.randint(0, 5)
    width = 0.35  # the width of the bars
    # ind = [i + width for i in range(g_MAX_ITERATION)]  # the x locations for the groups
    ind = range(g_MAX_ITERATION)[:]  # the x locations for the groups
    fig, ax = plt.subplots()
    rects = ax.bar(ind, menMeans, width, color='r', yerr=menStd)
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Number')
    ax.set_xticks(ind)
    plt.show()
    print "Learning finished"
    # Learned path to goal
    Agent.randomness = 0.0
    agent.current_state = init_state
    his = agent.start()
    # Show learned path to goal
    all_data = [(tpl[1], tpl[0]) for tpl in his[:]]
    plt.axis([-1, 20, -1, 20])
    plt.plot(*zip(*itertools.chain.from_iterable(itertools.combinations(all_data, 1))), color='brown', marker='o')
    plt.show()
