import numpy as np
import random
import tensorflow as tf
import os
import pickle
import time

from collections import deque

class ModelController(object):
    def __init__(self, observation_shape,
                       num_actions,
                       session=None,
                       random_action_probability=0.01,  #was 0.05
                       exploration_period=1000,
                       store_every_nth=5,
                       train_every_nth=5,
                       minibatch_size=32,
                       discount_rate=0.95,
                       max_experience=30000,
                       target_network_update_rate=0.01,
                       summary_writer=None):
        """Initialized the Deepq object.

        Based on:
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

        Parameters
        -------
        observation_shape : int
            length of the vector passed as observation
        num_actions : int
            number of actions that the model can execute
        observation_to_actions: dali model
            model that implements activate function
            that can take in observation vector or a batch
            and returns scores (of unbounded values) for each
            action for each observation.
            input shape:  [batch_size] + observation_shape
            output shape: [batch_size, num_actions]
        optimizer: tf.solver.*
            optimizer for prediction error
        session: tf.Session
            session on which to execute the computation
        random_action_probability: float (0 to 1)
        exploration_period: int
            probability of choosing a random
            action (epsilon form paper) annealed linearly
            from 1 to random_action_probability over
            exploration_period
        store_every_nth: int
            to further decorrelate samples do not all
            transitions, but rather every nth transition.
            For example if store_every_nth is 5, then
            only 20% of all the transitions is stored.
        train_every_nth: int
            normally training_step is invoked every
            time action is executed. Depending on the
            setup that might be too often. When this
            variable is set set to n, then only every
            n-th time training_step is called will
            the training procedure actually be executed.
        minibatch_size: int
            number of state,action,reward,newstate
            tuples considered during experience reply
        dicount_rate: float (0 to 1)
            how much we care about future rewards.
        max_experience: int
            maximum size of the reply buffer
        target_network_update_rate: float
            how much to update target network after each
            iteration. Let's call target_network_update_rate
            alpha, target network T, and network N. Every
            time N gets updated we execute:
                T = (1-alpha)*T + alpha*N
        summary_writer: tf.train.SummaryWriter
            writer to log metrics
        """
        # memorize arguments
        self.observation_shape         = observation_shape
        self.num_actions               = num_actions
        self.s                         = session

        self.random_action_probability = random_action_probability
        self.exploration_period        = exploration_period
        self.store_every_nth           = store_every_nth
        self.train_every_nth           = train_every_nth
        self.minibatch_size            = minibatch_size
        self.discount_rate             = tf.constant(discount_rate)
        self.max_experience            = max_experience
        self.target_network_update_rate = \
                tf.constant(target_network_update_rate)

        # deepq state
        self.actions_executed_so_far = 0
        self.experience = deque()

        self.iteration = 0
        self.summary_writer = summary_writer



    def linear_annealing(self, n, total, p_initial, p_final):
        """Linear annealing between p_initial and p_final
        over total steps - computes value at step n"""
        if n >= total:
            return p_final
        else:
            return p_initial - (n * (p_initial - p_final)) / (total)


    def observation_batch_shape(self, batch_size):
        return tuple([batch_size] + list(self.observation_shape))

    def create_variables(self):
        return 0

    def action(self, observation):
        """Given observation returns the action that should be chosen using
        DeepQ learning strategy. Does not backprop."""

        
        actions = np.zeros(len(observation[:,1]))
        for i in xrange(len(actions)):
            noise = random.random()
            if noise < .03:
                actions[i] = random.randint(0,1)
                continue
            assert observation[i].shape == self.observation_shape, \
                    "Action is performed based on single observation."

            self.actions_executed_so_far += 1
            exploration_p = self.linear_annealing(self.actions_executed_so_far,
                                                  self.exploration_period,
                                                  1.0,
                                                  self.random_action_probability)

            #looking for the closest object    
            direction = np.argmin(observation[i,:-4:5])   #5 and somewhat -4 are hardcoded in, ideally should be changed

            if direction < .5 * len(observation[i,:-4:5]):
                actions[i] = 0
            else:
                actions[i] = 1 

        return [actions, self.actions_executed_so_far]

    def exploration_completed(self):
        return min(float(self.actions_executed_so_far) / self.exploration_period, 1.0)

    def store(self, observation, action, reward, newobservation):
        """Store experience, where starting with observation and
        execution action, we arrived at the newobservation and got thetarget_network_update
        reward reward

        If newstate is None, the state/action pair is assumed to be terminal
        """
        if self.number_of_times_store_called % self.store_every_nth == 0:
            for i in xrange(len(reward)):
                self.experience.append((observation[i], action[i], reward[i], newobservation[i]))
                if len(self.experience) > self.max_experience:
                    self.experience.popleft()
        self.number_of_times_store_called += 1

    def training_step(self):
        """Pick a self.minibatch_size exeperiences from reply buffer
        and backpropage the value function.
        """
        if self.number_of_times_train_called % self.train_every_nth == 0:
            if len(self.experience) <  self.minibatch_size:
                return

            # sample experience.
            samples   = random.sample(range(len(self.experience)), self.minibatch_size)
            samples   = [self.experience[i] for i in samples]

            # bach states
            states         = np.empty(self.observation_batch_shape(len(samples)))
            newstates      = np.empty(self.observation_batch_shape(len(samples)))
            action_mask    = np.zeros((len(samples), self.num_actions))

            newstates_mask = np.empty((len(samples),))
            rewards        = np.empty((len(samples),))

            for i, (state, action, reward, newstate) in enumerate(samples):
                states[i] = state
                action_mask[i] = 0
                action_mask[i][int(action)] = 1
                rewards[i] = reward
                if newstate is not None:
                    newstates[i] = newstate
                    newstates_mask[i] = 1
                else:
                    newstates[i] = 0
                    newstates_mask[i] = 0

            states = states.astype(np.float32)
            
            calculate_summaries = self.iteration % 100 == 0 and \
                    self.summary_writer is not None

            cost, _, summary_str = self.s.run([
                self.prediction_error,
                self.train_op,
                self.summarize if calculate_summaries else self.no_op1,
            ], {
                self.observation:            states,
                self.next_observation:       newstates,
                self.next_observation_mask:  newstates_mask,
                self.action_mask:            action_mask,
                self.rewards:                rewards,
            })

            self.s.run(self.target_network_update)

            if calculate_summaries:
                self.summary_writer.add_summary(summary_str, self.iteration)

            self.iteration += 1

        self.number_of_times_train_called += 1

    def save(self, save_dir, debug=False):
        STATE_FILE      = os.path.join(save_dir, 'deepq_state')
        MODEL_FILE      = os.path.join(save_dir, 'model')

        # deepq state
        state = {
            'actions_executed_so_far':      self.actions_executed_so_far,
            'iteration':                    self.iteration,
            'number_of_times_store_called': self.number_of_times_store_called,
            'number_of_times_train_called': self.number_of_times_train_called,
        }

        if debug:
            print('Saving model... ',)

        saving_started = time.time()

        self.saver.save(self.s, MODEL_FILE)
        with open(STATE_FILE, "wb") as f:
            pickle.dump(state, f)

        print('done in {} s'.format(time.time() - saving_started))

    def restore(self, save_dir, debug=False):
        # deepq state
        STATE_FILE      = os.path.join(save_dir, 'deepq_state')
        MODEL_FILE      = os.path.join(save_dir, 'model')

        with open(STATE_FILE, "rb") as f:
            state = pickle.load(f)
        self.saver.restore(self.s, MODEL_FILE)

        self.actions_executed_so_far      = state['actions_executed_so_far']
        self.iteration                    = state['iteration']
        self.number_of_times_store_called = state['number_of_times_store_called']
        self.number_of_times_train_called = state['number_of_times_train_called']

    
    def kill_session(self):
        if self.s is not None:
             self.s.close()



