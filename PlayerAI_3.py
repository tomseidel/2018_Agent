from random import randint
from BaseAI_3 import BaseAI
from Grid_3       import Grid
import time
import math

timeLimit = 0.2
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)


class FeatureTransformer:
    def __init__(self, env):
        observation_examples = np.random.random((20000, 4))*2 - 1 
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
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer

        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n, D))
        for i in range(env.action_space.n):
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

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

# returns a list of states_and_rewards, and the total reward
def play_one(model, eps, gamma, lambda_):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
    # while not done and iters < 200:
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, info = env.step(action)

        # update the model
        G = reward + gamma*np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G, gamma, lambda_)

        totalreward += reward
        iters += 1

    return totalreward

class PlayerAI(BaseAI):

    def __init__(self):
        self.allCutoffsMade = 0
        
    def getMove(self, grid):
        
        moves = grid.getAvailableMoves()
        return moves[randint(0, len(moves) - 1)] if moves else None
        