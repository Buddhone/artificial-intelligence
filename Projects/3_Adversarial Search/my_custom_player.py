
from sample_players import DataPlayer
import math
import random


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        depth_limit = 10
        if state.ply_count ==0:
            self.queue.put(random.choice(state.actions()))
        else:
            for depth in range(1, depth_limit + 1):
               action = self.alpha_beta(state, depth)
               if action is not None:
                   self.queue.put(action)
              
    def minimax_decision(self, state, depth):
                
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            v = self.min_value(state.result(a), depth - 1)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move
    
    def alpha_beta(self, state, depth):
        alpha =float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            v = self.min_value(state.result(a), best_score, beta, depth - 1)
            if v > best_score:
                best_score = v
                best_move = a
    
        return best_move
        
    def min_value(self, state, alpha, beta, depth):
        
        if depth <= 0:
            return self.heuristic3(state)        # Switch score with heuristic1 / heuristic2 / heuristic3
        if state.terminal_test():
            return state.utility(self.player_id)
        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), alpha, beta, depth-1))
            if v <= alpha:
                return v
            beta = min(beta, v)            
        return v

    def max_value(self, state, alpha, beta, depth):
        if depth <= 0:
            return self.heuristic3(state)       # Switch score with heuristic1 and heuristic2
        if state.terminal_test():
            return state.utility(self.player_id)            
        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), alpha, beta, depth-1))
            if v >= beta:
                return v
            alpha = max(alpha, v)            
        return v
  
    def score(self, state):
        # # own moves - opponent moves heuristic
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    
    def heuristic1(self, state):   
        """
        This heuristic will be more aggressive than heuristic1, the player distance add aggressiveness
        """
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_moves_minus_opp_moves = self.score(state)
        own_distance_from_center= self.distance_from_center(self.coordinates(own_loc))
        return own_moves_minus_opp_moves - own_distance_from_center
    
    def heuristic2(self, state):   
        """
        This heuristic will be more aggressive than heuristic1 the player distance reduces aggressiveness
        """
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        players_distance = self.euclidean_distance(self.coordinates(own_loc), self.coordinates(opp_loc))
        own_moves_minus_opp_moves = self.score(state)
        own_distance_from_center= self.distance_from_center(self.coordinates(own_loc))
        return own_moves_minus_opp_moves - own_distance_from_center - players_distance
    
    def heuristic3(self, state):   
        """
        This heuristic will be more aggressive than heuristic1 the player distance reduces aggressiveness
        """
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        players_distance = self.euclidean_distance(self.coordinates(own_loc), self.coordinates(opp_loc))
        own_moves_minus_opp_moves = self.score(state)
        own_distance_from_center= self.distance_from_center(self.coordinates(own_loc))
        return own_moves_minus_opp_moves - own_distance_from_center + players_distance
    
    def coordinates(self, encoded_location):
        """
        Gets x,y coordinates out of the encoded location
        """
        x = encoded_location % 13 
        y = math.floor(encoded_location/13) 
        return x, y

    def distance_from_center(self, player_location):
        """
        Calculating the Euclidean distance to centre
        """
        centre_location= 5, 4
        return self.euclidean_distance(player_location, centre_location)
    
    def euclidean_distance(self, loc1, loc2):
        """
        Returns the manhattan distance between two points (loc1 and loc2)
        """
        dx = abs(loc1[0] - loc2[0])
        dy = abs(loc1[1] - loc2[1])
        return math.sqrt(dx * dx + dy * dy)
