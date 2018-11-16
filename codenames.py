import random
import re
import numpy as np
import math
import csv
import os
from nltk.stem import PorterStemmer
import ngrams
import util
#from search import CodenamesSearchProblem

from typing import List, Tuple, Iterable

# This file stores the "solutions" the bot had intended,
# when you play as agent and the bot as spymaster.
log_file = open("log_file", "w")

clue_count = []
wrong_guesses = 0
turns_taken = 0
RESULTS_FILE_PATH = "results.csv"

def log_result(score, clue_count, turns_taken, wrong_guesses):
    # log result of the game
    new_file = False
    if (not os.path.isfile(RESULTS_FILE_PATH)):
        new_file = True
        
    with open (RESULTS_FILE_PATH, 'a') as f:
        writer = csv.writer(f)
        if (new_file):
            writer.writerow(['Final score', 'Clue Count', 'Sum Clue Count', 'Turns Taken', 'Wrong Guesses'])
        result = [score, clue_count, sum(clue_count), turns_taken, wrong_guesses]
        writer.writerow(result)

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
            #actions.append(("%s for group %s" % (clue, group), (new_board, new_words), self.cost(clue, group)))
            actions.append( ((clue, group), (new_board, new_words), self.cost(clue, group)) )
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

class Reader:
    def read_picks(
        self, words, my_words, guesses_left) -> List[str]:
        """
        Query the user for guesses.
        :param words: Words the user can choose from.
        :param my_words: Correct words.
        :param guesses_left: Number of guesses left
        :return: The words picked by the user.
        """
        raise NotImplementedError

    def read_clue(self, word_set: Iterable[str]) -> Tuple[str, int]:
        """
        Read a clue from the (spymaster) user.
        :param word_set: Valid words
        :return: The clue and number given.
        """
        raise NotImplementedError

    def print_words(self, words: List[str], nrows: int):
        """
        Prints a list of words as a 2d table, using `nrows` rows.
        :param words: Words to be printed.
        :param nrows: Number of rows to print.
        """
        raise NotImplementedError


class TerminalReader(Reader):
    def read_picks(
        self, words: List[str], my_words: Iterable[str], guesses_left: int) -> List[str]:
        global wrong_guesses
        print("Guesses left:", guesses_left)
        print("Remaining words:",  len(my_words))
        #TODO: replace input with call to neural network
        guess = input("Your guess: ").strip().lower()
        if guess in my_words:
            print("Correct!")
        else:
            wrong_guesses += 1
            print("Wrong :(")
        return guess

    def read_clue(self, word_set) -> Tuple[str, int]:
        while True:
            inp = input("Clue (e.g. 'car 2'): ").lower()
            match = re.match("(\w+)\s+(\d+)", inp)
            if match:
                clue, cnt = match.groups()
                if clue not in word_set:
                    print("I don't understand that word.")
                    continue
                return clue, int(cnt)

    def print_words(self, words: List[str], nrows: int):
        longest = max(map(len, words))
        print()
        for row in zip(*(iter(words),) * nrows):
            for word in row:
                print(word.rjust(longest), end=" ")
            print()
        print()


class Codenames:
    def __init__(self, cnt_rows=5, cnt_cols=5, cnt_agents=8, agg=.6):
        """
        :param cnt_rows: Number of rows to show.
        :param cnt_cols: Number of columns to show.
        :param cnt_agents: Number of good words.
        :param agg: Agressiveness in [0, infinity). Higher means more aggressive.
        """
        self.cnt_rows = cnt_rows
        self.cnt_cols = cnt_cols
        self.cnt_agents = cnt_agents
        self.agg = agg

        # Other
        self.vectors = np.array([])
        self.word_list = []
        self.weirdness = []
        self.word_to_index = {}
        self.codenames = []

    def load(self, datadir):
        # Glove word vectors
        print("...Loading vectors")
        #self.vectors = np.load("%s/glove.6B.300d.npy" % datadir)
        self.vectors = np.load('kirkby_vectors.npy')
        # self.vectors = np.load('fitted_vectors/glove_fitted.npy')
        # self.vectors = np.load('fitted_vectors/kirkby_fitted.npy') # not good
        # self.vectors = np.load('fitted_vectors/glove_with_ontology.npy') # not great
        # self.vectors = np.load('fitted_vectors/kirkby_with_ontology.npy') # super bad

        # List of all glove words
        print("...Loading words")
        # self.word_list = [w.lower().strip() for w in open("%s/words" % datadir)]
        # self.word_list = [w.lower().strip() for w in open("fitted_vectors/glove_fitted_words.txt")]
        # self.word_list = [w.lower().strip() for w in open("fitted_vectors/kirkby_fitted_words.txt")]
        self.word_list = [w.lower().strip() for w in open("kirkby_wv.txt")]
        # print("wordlist:", self.word_list)
        self.weirdness = [math.log(i + 1) + 1 for i in range(len(self.word_list))]

        # Indexing back from word to indices
        print("...Making word to index dict")
        self.word_to_index = {w: i for i, w in enumerate(self.word_list)}

        # All words that are allowed to go onto the table
        print("...Loading codenames")
        self.codenames = [
            word
            for word in (w.lower().strip().replace(" ", "-") for w in open("wordlist2"))
            if word in self.word_to_index
        ]

        print("Ready!")

    def word_to_vector(self, word: str) -> np.ndarray:
        """
        :param word: To be vectorized.
        :return: The vector.
        """
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0: 
               return v
            return v / norm
        if word == "---":
            return np.zeros(self.vectors[0].shape)
        return normalize(self.vectors[self.word_to_index[word]])

    def most_similar_to_given(self, clue: str, choices: List[str]) -> str:
        """
        :param clue: Clue from the spymaster.
        :param choices: Choices on the table.
        :return: Which choice to go for.
        """
        clue_vector = self.word_to_vector(clue)
        return max(choices, key=lambda w: self.word_to_vector(w) @ clue_vector)

    def find_clue(
        self, words: List[str], my_words: List[str]) -> Tuple[str, float, List[str]]:
        """
        :param words: Words on the board.
        :param my_words: Words we want to guess.
        :return: (The best clue, the score, the words we expect to be guessed)
        """
        print("Thinking", end="", flush=True)

        # Words to avoid the agent guessing.
        negs = [w for w in words if w not in my_words]
        # Worst (highest) inner product with negative words
        nm = (
            self.vectors @ np.array([self.word_to_vector(word) for word in negs]).T
        ).max(axis=1)
        # Inner product with positive words
        pm = self.vectors @ np.array([self.word_to_vector(word) for word in my_words]).T

        best_clue, best_score, best_k, best_g = None, -1, 0, ()
        for step, (clue, lower_bound, scores) in enumerate(zip(self.word_list, nm, pm)):

            # print("words:", clue, lower_bound, scores)
            
            if step % 20000 == 0:
                print(".", end="", flush=True)

            # If the best score is lower than the lower bound, there is no reason
            # to even try it.
            ps = PorterStemmer()
            stem = ps.stem(clue)

            prob = ngrams.Pwords([clue.lower()])
            if prob < 1e-12:
                continue
            if max(scores) <= lower_bound or stem in words or clue in self.blacklist or stem in self.blacklist:
                continue

            # Order scores by lowest to highest inner product with the clue.
            ss = sorted((s, i) for i, s in enumerate(scores))
            # Calculate the "real score" by
            #    (lowest score in group) * [ (group size)^aggressiveness - 1].
            # The reason we subtract one is that we never want to have a group of
            # size 1.
            # We divide by log(step), as to not show too many 'weird' words.
            real_score, j = max(
                (
                    (s - lower_bound)
                    * ((len(ss) - j) ** self.agg - .99)
                    # * -0.1 * math.log(prob)
                    / self.weirdness[step],
                    j,
                )
                for j, (s, _) in enumerate(ss)
            )

            if real_score > best_score:
                group = [my_words[i] for _, i in ss[j:]]
                best_clue, best_score, best_k, best_g = (
                    clue,
                    real_score,
                    len(group),
                    group,
                )

            # print("best score:", best_score)
            # print("best clue:", best_clue)
            # print("group:", best_g)

        # After printing '.'s with end="" we need a clean line.
        print()

        return best_clue, best_score, best_g

    def generate_start_state(self):
        words = random.sample(self.codenames, self.cnt_rows * self.cnt_cols)
        my_words = set(random.sample(words, self.cnt_agents))
        self.blacklist = set(my_words)
        return words, my_words

    def save_train_example(self, board, guess):
        # add the training example to the csv
        train_examples_csv = os.path.join(TRAIN_DATA_DIR, 'train_examples.csv')
        with open(train_examples_csv, "a") as f:
            writer = csv.writer(f)
            writer.writerow([guess] + board)

        #train_examples_wordvec_csv = os.path.join(TRAIN_DATA_DIR, 'train_examples_wordvec.csv')
        #with open(train_examples_wordvec_csv, "a") as f:
        #    train_ex_matrix = self.word_to_vector(guess)
        #    print(train_ex_matrix.shape)
        #    for word in board:
        #        train_ex_matrix = np.vstack((train_ex_matrix, self.word_to_vector(word)))

        #    np.savetxt(f, train_ex_matrix)

    def play_spymaster(self, reader: Reader):
        """
        Play a complete game, with the robot being the spymaster.
        """
        global clue_count, wrong_guesses, turns_taken
        clue_count = []
        wrong_guesses = 0
        turns_taken = 0

        words, my_words = self.generate_start_state()

        while my_words:
            actions, costs = find_next_clue(tuple(words), tuple(my_words), self)
            clue, group = actions[0]
            #clue, score, group = self.find_clue(words, list(my_words))
            # Print the clue to the log_file for "debugging" purposes
            group_scores = np.array(
                [self.word_to_vector(w) for w in group]
            ) @ self.word_to_vector(clue)
            print(clue, group, group_scores, file=log_file, flush=True)
            # Save the clue, so we don't use it again
            self.blacklist.add(clue)

            clue_count.append(len(group))
            turns_taken += 1

            print()
            print(
                'Clue: "{} {}" (certainty {:.2f}, remaining words {})'.format(
                    clue, len(group), 0, len(my_words)
                )
            )
            #print(
            #    'Clue: "{} {}" (certainty {:.2f}, remaining words {})'.format(
            #        clue, len(group), score, len(my_words)
            #    )
            #)
            print()
            guesses_left = len(group)
            while guesses_left:
                reader.print_words(words, nrows=self.cnt_rows)
                pick = reader.read_picks(words, my_words, guesses_left)
                if pick not in words:
                    print("That isn't an option. Please try again!")
                    continue

                # off board (input) and pick (label)
                # self.save_train_example(words, pick)

                words[words.index(pick)] = "---"
                if pick in my_words:
                    my_words.remove(pick)
                    guesses_left -= 1
                else:
                    guesses_left = 0

            # for pick in reader.read_picks(words, my_words, len(group)):
            #     words[words.index(pick)] = "---"
            #     if pick in my_words:
            #         my_words.remove(pick)

        score = 5 / (sum(clue_count) / len(clue_count)) + turns_taken + 2 * wrong_guesses
        print("final score:", score)

        log_result(score, clue_count, turns_taken, wrong_guesses)

    def play_agent(self, reader: Reader):
        """
        Play a complete game, with the robot being the agent.
        """
        words = random.sample(self.codenames, self.cnt_rows * self.cnt_cols)
        my_words = random.sample(words, self.cnt_agents)
        picked = []
        while any(w not in picked for w in my_words):
            reader.print_words(
                [w if w not in picked else "---" for w in words], nrows=self.cnt_rows
            )
            print("Your words:", ", ".join(w for w in my_words if w not in picked))
            clue, cnt = reader.read_clue(self.word_to_index.keys())
            for _ in range(cnt):
                guess = self.most_similar_to_given(
                    clue, [w for w in words if w not in picked]
                )
                picked.append(guess)
                answer = input("I guess {}? [Y/n]: ".format(guess))
                if answer == "n":
                    print("Sorry about that.")
                    break
            else:
                print("I got them all!")

def main():
    cn = Codenames()
    cn.load("dataset")
    reader = TerminalReader()
    while True:
        try:
            mode = input("\nWill you be agent or spymaster?: ")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        try:
            if mode == "spymaster":
                cn.play_agent(reader)
            elif mode == "agent":
                cn.play_spymaster(reader)
        except KeyboardInterrupt:
            # Catch interrupts from play functions
            pass


main()
