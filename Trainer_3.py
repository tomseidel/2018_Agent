from Grid_3       import Grid
from ComputerAI_3 import ComputerAI
from PlayerAI_3   import PlayerAI
from Displayer_3  import Displayer
from random       import randint
import time
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
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

class FeatureTransformer:
    def __init__(self):
        observation_examples = np.random.random((20000, 16))*2 - 1 
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=500))
                ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        # print "observations:", observations
        scaled = self.scaler.transform(observations)
        # assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)
 
class BaseModel:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, input_, target, eligibility, lr=10e-3):
        self.w += lr*(target - input_.dot(self.w))*eligibility

    def predict(self, X):
        X = np.array(X)
        return X.dot(self.w)

# Holds one BaseModel for each action
class Model:
    def __init__(self, feature_transformer):
        self.models = []
        self.feature_transformer = feature_transformer

        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((len(actionDic), D))
        for i in range(len(actionDic)):
            model = BaseModel(D)
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G, gamma, lambda_):
        X = self.feature_transformer.transform([s])
        assert(len(X.shape) == 2)
        self.eligibilities *= gamma*lambda_
        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X[0], G, self.eligibilities[a])

    def sample_action(self, s, eps, grid):
        moves = grid.getAvailableMoves()
            
        if np.random.random() < eps:
            #return self.env.action_space.sample()
            #from available actions
            return moves[randint(0, len(moves) - 1)] if moves else None
        else:
            nextMove = np.argmax(self.predict(s))
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


    def play_one(self, model, eps, gamma, lambda_):
        for i in range(self.initTiles):
            self.insertRandonTile()

        observation = np.array(self.grid.map).ravel()
        done = False
        totalreward = 0
        iters = 0
  
        ##self.displayer.display(self.grid)

        # Player AI Goes First
        turn = PLAYER_TURN
        maxTile = 0

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
                
                ##print(actionDic[move])

                # Validate Move
                if move != None and move >= 0 and move < 4:
                    if self.grid.canMove([move]):
                        self.grid.move(move)

                        #observation, reward, done, info = env.step(action)
                        observation = np.array(self.grid.map).ravel() #get new observation
                        gridNewAvg = self.gridAverage(self.grid)
                        #newMax = self.grid.getMaxTile()
                        reward = gridNewAvg - gridAvg
                        #reward = newMax - lastMax

                        # update the model
                        G = reward + gamma*np.max(model.predict(observation)[0])
                        model.update(prev_observation, move, G, gamma, lambda_)

                        totalreward += reward
                        iters += 1
                        
                        # Update maxTile
                        maxTile = self.grid.getMaxTile()
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
    
    

    #gameManager.start()
    ft = FeatureTransformer()
    model = Model(ft)
    gamma = 0.99
    lambda_ = 0.7

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
    
        # eps = 1.0/(0.1*n+1)
        eps = 0.5*(0.99**n)
        #eps = 0.5/np.sqrt(n+1)
        totalreward = gameManager.play_one(model, eps, gamma, lambda_)
        totalrewards[n] = totalreward
        #print("episode:", n, "total reward:", totalreward, "epsilon:", eps)

    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())
    

if __name__ == '__main__':
    main()
