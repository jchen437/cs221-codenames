import numpy as np
import util
import codenames
from nltk.stem import PorterStemmer
import ngrams

class CodenamesSearchProblem(util.SearchProblem):
    
    def __init__(self, board, my_words, game):
        self.board = board
        self.my_words = my_words
        self.game = game # instance of CodeNames class

    def startState(self):
        # board, words left to guess
        return (self.board, self.my_words)

    # Return whether |state| is an end state or not.
    def isEnd(self, state):
        board, my_words = state
        return not board or not my_words

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state):
        board, my_words = state
        actions = []
        for clue, group in self.generate_poss_clues(board, my_words):
            new_board = tuple([w for w in board if w not in group])
            new_words = tuple([w for w in my_words if w not in group])
            actions.append(("%s for group %s" % (clue, group), (new_board, new_words), self.cost(clue, group)))
        return actions

    def generate_poss_clues(self, board, my_words):
        negs = [w for w in board if w not in my_words]
        nm = (
            self.game.vectors @ np.array([self.game.word_to_vector(word) for word in negs]).T
        ).max(axis=1)
        pm = self.game.vectors @ np.array([self.game.word_to_vector(word) for word in my_words]).T
        clue_groups = []
        for step, (clue, lower_bound, scores) in enumerate(zip(self.game.word_list, nm, pm)):

            ps = PorterStemmer()
            stem = ps.stem(clue)

            prob = ngrams.Pwords([clue.lower()])
            if prob < 1e-12:
                continue
            if max(scores) <= lower_bound or stem in board or clue in self.game.blacklist or stem in self.game.blacklist:
                continue

            ss = sorted((s, i) for i, s in enumerate(scores))

            real_score, j = max(
                (
                    (s - lower_bound)
                    * ((len(ss) - j) ** self.game.agg - .99)
                    / self.game.weirdness[step],
                    j,
                )
                for j, (s, _) in enumerate(ss)
            )

            group = [my_words[i] for _, i in ss[j:]]
            clue_groups.append((clue, group))

        return clue_groups

    def cost(self, clue, group):
        clue_vec = self.game.word_to_vector(clue)
        cost = 0
        for word in group:
            word_vec = self.game.word_to_vector(word)
            cost -= np.dot(word_vec.T, clue_vec)
        return cost


def find_next_clue(board, my_words, game):
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(CodenamesSearchProblem(board, my_words, game))

    return ucs.actions, ucs.totalCost

def main():
    cn = codenames.Codenames()
    cn.load("dataset")
    
    board, my_words = cn.generate_start_state()
    print("board:", board)
    print("words to guess:", my_words)

    actions, cost = find_next_clue(tuple(board), tuple(my_words), cn)

    print("Actions:")
    for a in actions:
        print(a)

    print("Cost:", cost)

main()