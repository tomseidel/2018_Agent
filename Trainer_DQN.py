from Grid_3       import Grid
from ComputerAI_3 import ComputerAI
from PlayerAI_3   import PlayerAI
from Displayer_3  import Displayer
from random       import randint
import time
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import normalize
import tensorflow as tf


defaultInitialTiles = 2
defaultProbability = 0.9

actionDic = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

(PLAYER_TURN, COMPUTER_TURN) = (0, 1)

# Time Limit Before Losing
timeLimit = 0.2
allowance = 0.05

# a version of HiddenLayer that keeps track of params
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.sigmoid, use_bias=True):
    self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
      self.params.append(self.b)
    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)


class DQN:
  def __init__(self, D, K, hidden_layer_sizes, gamma, max_experiences=100000, min_experiences=100, batch_sz=64):
    self.K = K
    #self.scaler = StandardScaler()
    self.scaler = MinMaxScaler((0,2048))

    # create the graph
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, K, lambda x: x)
    self.layers.append(layer)

    # collect params for copy
    self.params = []
    for layer in self.layers:
      self.params += layer.params

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

    # calculate output and cost
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = Z
    self.predict_op = Y_hat

    selected_action_values = tf.reduce_sum(
      Y_hat * tf.one_hot(self.actions, K),
      reduction_indices=[1]
    )

    cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
    #self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)
    # self.train_op = tf.train.AdagradOptimizer(10e-3).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(10e-4, momentum=0.9).minimize(cost)
    self.train_op = tf.train.GradientDescentOptimizer(10e-5).minimize(cost)

    # create replay memory
    self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
    self.max_experiences = max_experiences
    self.min_experiences = min_experiences
    self.batch_sz = batch_sz
    self.gamma = gamma

  def set_session(self, session):
    self.session = session

  def copy_from(self, other):
    # collect all the ops
    ops = []
    my_params = self.params
    other_params = other.params
    for p, q in zip(my_params, other_params):
      actual = self.session.run(q)
      op = p.assign(actual)
      ops.append(op)
    # now run them all
    self.session.run(ops)

  def predict(self, X):
    X = np.atleast_2d(X)

    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def train(self, target_network):
    # sample a random batch from buffer, do an iteration of GD
    if len(self.experience['s']) < self.min_experiences:
      # don't do anything if we don't have enough experience
      return

    # randomly select a batch
    idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False)
    # print("idx:", idx)
    states = [self.experience['s'][i] for i in idx]
    actions = [self.experience['a'][i] for i in idx]
    rewards = [self.experience['r'][i] for i in idx]
    next_states = [self.experience['s2'][i] for i in idx]
    dones = [self.experience['done'][i] for i in idx]
    next_Q = np.max(target_network.predict(next_states), axis=1)
    targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

    # call optimizer
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )

  def add_experience(self, s, a, r, s2, done):
    if len(self.experience['s']) >= self.max_experiences:
      self.experience['s'].pop(0)
      self.experience['a'].pop(0)
      self.experience['r'].pop(0)
      self.experience['s2'].pop(0)
      self.experience['done'].pop(0)
    self.experience['s'].append(s)
    self.experience['a'].append(a)
    self.experience['r'].append(r)
    self.experience['s2'].append(s2)
    self.experience['done'].append(done)

  def sample_action(self, x, eps, grid):
    moves = grid.getAvailableMoves()
    if np.random.random() < eps:
      #return np.random.choice(self.K)
      return moves[randint(0, len(moves) - 1)] if moves else None
    else:
      X = np.atleast_2d(x)
      nextMove = np.argmax(self.predict(X)[0])
      if (nextMove in moves):
        return nextMove
      else:
        return moves[randint(0, len(moves) - 1)] if moves else None
      


class GameManager:
    def __init__(self, size = 4):
        self.grid = Grid(size)
        self.possibleNewTiles = [2, 4]
        self.probability = defaultProbability
        self.initTiles  = defaultInitialTiles
        self.computerAI = None
        self.playerAI   = None
        self.displayer  = None
        self.over       = False

    def setComputerAI(self, computerAI):
        self.computerAI = computerAI

    def setPlayerAI(self, playerAI):
        self.playerAI = playerAI

    def setDisplayer(self, displayer):
        self.displayer = displayer

    def updateAlarm(self, currTime):
        if currTime - self.prevTime > timeLimit + allowance:
            self.over = True
        else:
            while time.clock() - self.prevTime < timeLimit + allowance:
                pass

            self.prevTime = time.clock()

    def gridAverage(self, grid):
        cells = []

        for x in range(4):
            for y in range(4):
                if grid.map[x][y] > 0:
                    cells.append((x,y))
        sum = 0
        for c in cells:
            sum += grid.map[c[0]][c[1]]
        return sum / len(cells)


    def play_one(self, model, tmodel, eps, gamma, copy_period):
        for i in range(self.initTiles):
            self.insertRandonTile()

        grid_array = np.array(self.grid.map).ravel()
        #observation = grid_array / np.linalg.norm(grid_array)
        observation = grid_array / 2048
        
        done = False
        totalreward = 0
        iters = 0
  
        ##self.displayer.display(self.grid)

        # Player AI Goes First
        turn = PLAYER_TURN
        maxTile = 2

        self.prevTime = time.clock()

        #while not self.isGameOver() and not self.over:
        while not self.isGameOver():
            # Copy to Ensure AI Cannot Change the Real Grid to Cheat
            gridCopy = self.grid.clone()

            move = None

            if turn == PLAYER_TURN:
                ##print("Player's Turn:", end="")
                #move = self.playerAI.getMove(gridCopy)
                move = model.sample_action(observation, eps, gridCopy)
                gridAvg = self.gridAverage(gridCopy)
                #lastMax = gridCopy.getMaxTile()
                prev_observation = observation
                last_maxTile = maxTile
                
                ##print(actionDic[move])

                # Validate Move
                if move != None and move >= 0 and move < 4:
                    if self.grid.canMove([move]):
                        self.grid.move(move)
                        maxTile = self.grid.getMaxTile()
                        
                        #observation, reward, done, info = env.step(action)
                        grid_array = np.array(self.grid.map).ravel()
                        #observation = grid_array / np.linalg.norm(grid_array)
                        observation = grid_array / maxTile
        
                        gridNewAvg = self.gridAverage(self.grid)
                        #newMax = self.grid.getMaxTile()
                        reward = gridNewAvg - gridAvg
                        
                        # Update maxTile
                        #reward = maxTile - last_maxTile
                        #reward = 1
                        
                        done = self.isGameOver()
                        if done:
                            reward = -100

                        # update the model
                        model.add_experience(prev_observation, move, reward, observation, done)
                        model.train(tmodel)
                        
                        if iters % copy_period == 0:
                            tmodel.copy_from(model)
                        
                        if reward == 1:
                            totalreward += reward
                        iters += 1

                    else:
                        print("Invalid PlayerAI Move")
                        self.over = True
                else:
                    print("Invalid PlayerAI Move - 1")
                    self.over = True
            else:
                ##print("Computer's turn:")
                move = self.computerAI.getMove(gridCopy)

                # Validate Move
                if move and self.grid.canInsert(move):
                    self.grid.setCellValue(move, self.getNewTileValue())
                else:
                    print("Invalid Computer AI Move")
                    self.over = True

            #if not self.over:
                #self.displayer.display(self.grid)

            # Exceeding the Time Allotted for Any Turn Terminates the Game
            # self.updateAlarm(time.clock())

            turn = 1 - turn
        self.maxTile = self.grid.getMaxTile()
        print(maxTile)
        return maxTile

    def isGameOver(self):
        return not self.grid.canMove()

    def getNewTileValue(self):
        if randint(0,99) < 100 * self.probability:
            return self.possibleNewTiles[0]
        else:
            return self.possibleNewTiles[1];

    def insertRandonTile(self):
        tileValue = self.getNewTileValue()
        cells = self.grid.getAvailableCells()
        cell = cells[randint(0, len(cells) - 1)]
        self.grid.setCellValue(cell, tileValue)

def main():
    
    gamma = 0.9
    copy_period = 50

    D = 16 #??
    K = len(actionDic)
    sizes = [256,256]
    model = DQN(D, K, sizes, gamma)
    tmodel = DQN(D, K, sizes, gamma)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    model.set_session(session)
    tmodel.set_session(session)


    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        gameManager = GameManager()
        playerAI  	= PlayerAI()
        computerAI  = ComputerAI()
        displayer 	= Displayer()

        gameManager.setDisplayer(displayer)
        gameManager.setPlayerAI(playerAI)
        gameManager.setComputerAI(computerAI)
    
        #eps = 1.0/np.sqrt(n+1)
        eps = 0.1*(0.99**n)
        totalreward = gameManager.play_one(model, tmodel, eps, gamma, copy_period)
        totalrewards[n] = totalreward
        if n % 50 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 50):", totalrewards[max(0, n-50):(n+1)].mean())

        #print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
        #print("total steps:", totalrewards.sum())
    

if __name__ == '__main__':
    main()
