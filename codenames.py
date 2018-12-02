import random
import re
import numpy as np
import math
import csv
import os
from nltk.stem import PorterStemmer
import ngrams
import util
from sklearn.cluster import KMeans
from textblob import TextBlob
#from search import CodenamesSearchProblem

from typing import List, Tuple, Iterable

# This file stores the "solutions" the bot had intended,
# when you play as agent and the bot as spymaster.
log_file = open("log_file", "w")
game_log = open("game_log.txt", "a+")

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
        print(".", end="", flush=True)
        board, my_words = state
        actions = []
        for clue, group in self.generate_poss_clues(board, my_words):
            new_board = tuple([w for w in board if w not in group])
            new_words = tuple([w for w in my_words if w not in group])
            actions.append( ((clue, group), (new_board, new_words), self.cost(clue, group, board, my_words)) )
        return actions

    def generate_poss_clues(self, board, my_words):
        pm = self.game.vectors @ np.array([self.game.word_to_vector(word) for word in my_words]).T
        clue_groups = []
        for step, (clue, lower_bound, scores) in enumerate(zip(self.game.word_list, self.game.nm, pm)):

            prob = ngrams.Pwords([clue])
            if prob < 1e-10:
                continue

            ps = PorterStemmer()
            single = self.game.word_single[clue]
            if not single:
                continue
            stem = ps.stem(single)

            if max(scores) <= lower_bound or stem in self.game.stems or clue in self.game.blacklist or stem in self.game.blacklist:
                continue

            ss = sorted((s, i) for i, s in enumerate(scores))[-self.game.clue_cap:]

            real_score, j = max(
                (
                    (s - lower_bound)
                    * ((len(ss) - j) ** self.game.agg - .99),
                    # / self.game.weirdness[step],
                    j,
                )
                for j, (s, _) in enumerate(ss)
            )

            group = [my_words[i] for _, i in ss[j:]]
            clue_groups.append((real_score, (clue, group)))
        ##END
        return [p[1] for p in sorted(clue_groups)[-1000:]]
        # return clue_groups

    def cost(self, clue, group, board, my_words):
        negs = self.game.negs
        clue_vec = self.game.word_to_vector(clue)
        cost = 0
        for word in group:
            word_vec = self.game.word_to_vector(word)
            cost -= np.dot(word_vec.T, clue_vec)/(np.linalg.norm(word_vec.T) * np.linalg.norm(clue_vec))
        # worst = 0
        # for word in negs:
        #     neg_vec = self.game.word_to_vector(word)
        #     neg_cost =  np.dot(neg_vec.T, clue_vec)/(np.linalg.norm(word_vec.T) * np.linalg.norm(clue_vec))
        #     if neg_cost > worst:
        #        worst = neg_cost
        # cost += worst
        return cost


def print_to_txt(file, string):
    with open(file, "a") as f:
        f.write(string)

def find_next_clue_kmeans(board, my_words, game):
    #neg_vec = np.array([game.word_to_vector(word) for word in game.negs])
    pos_vec = np.array([game.word_to_vector(word) for word in my_words])
    # KAIS: emperically K=4 seems best if we just use a static K
    #kmeans = KMeans(n_clusters=min(len(my_words),3), random_state=0).fit(pos_vec)
    kmeans = None
    K = 4
    if len(my_words) > K:
        kmeans = KMeans(n_clusters=K, random_state=0).fit(pos_vec)
    else:
        scaled_K = math.ceil(len(my_words) / 2)
        kmeans = KMeans(n_clusters=scaled_K, random_state=0).fit(pos_vec)
    #kmeans = KMeans(n_clusters=min(len(my_words),4), random_state=0).fit(pos_vec)
    #kmeans = KMeans(n_clusters=min(len(my_words),5), random_state=0).fit(pos_vec)
    centers = kmeans.cluster_centers_

    closest  = float('-inf')
    closest_center_index = 0
    closest_word = None

    group = []
    avg_dist = [0] * len(centers)
    cluster_counts = [0] * len(centers)
    clusters = [[] for _ in range(len(centers))]
    # loop over all clusters, get avg dist to center, count the number of words too
    for i, label in enumerate(kmeans.labels_):
        avg_dist[label] += np.linalg.norm(centers[label] - pos_vec[i])
        cluster_counts[label] += 1
        clusters[label].append(my_words[i])

    for c, cluster in zip(cluster_counts, clusters):
        print_to_txt("cluster.txt", str(c) + " ")
        print_to_txt("cluster.txt", str(cluster) + "\n")
    
    for step, clue in enumerate(game.word_list):

        ps = PorterStemmer()
        tb = game.word_blobs[clue].words

        if not tb:
            continue

        stem = ps.stem(tb[0].singularize())

        prob = ngrams.Pwords([clue.lower()])
        if prob < 1e-12:
            continue
        if stem in game.stems or clue in game.blacklist or stem in game.blacklist:
            continue

        clue_vec = game.word_to_vector(clue)

        highest_neg_sim = game.clue_neg_sim[clue]

        highest = float('-inf')
        highest_center_index = 0
        highest_word = None

        # loop over all centers, find the cluster which is closest to the clue
        for i, center in enumerate(centers):
            # want similarity to be higher
            similarity = np.dot(clue_vec.T, center)/(np.linalg.norm(clue_vec.T) * np.linalg.norm(center))
            # weight = similarity with centroid * num words in cluster * avg dist - similarity with closest neg
            # we want to get the cluster the clue is closest to, want it to have more words and each
            # word should be close to the center, then we unweight this cluster by tis distance to the clue's
            # most similar neg word
            #print("sim: {}, count: {}, neg_sim: {}".format(similarity, cluster_counts[i], highest_neg_sim))
            weight = similarity * cluster_counts[i] - 0.1*highest_neg_sim #  - 0.1 * avg_dist[i], avg_dist needs to be subtracted but then results in the, of, and
            if weight > highest:
                highest = weight
                highest_center_index = i
                highest_word = clue

        if highest > closest:
            closest = highest
            closest_center_index = highest_center_index
            closest_word = highest_word

    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == closest_center_index:
            group.append(my_words[i])

    print_to_txt("cluster.txt", "clue: {}, group: {}\n".format(closest_word, group))
    return closest_word, group

def find_next_clue(board, my_words, game):

    ucs = util.UniformCostSearch(verbose=0)
    print("Thinking", end="", flush=True)
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
        guess = input("Your guess: ").strip().lower()
        if guess in my_words:
            print("Correct!")
            print("Guess: %s - correct" % guess, file=game_log, flush=True)
        else:
            wrong_guesses += 1
            print("Wrong :(")
            print("Guess: %s - not correct" % guess, file=game_log, flush=True)
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

    def print_words(self, words: List[str], nrows: int, to_file=None):
        longest = max(map(len, words))
        print("", file=to_file, flush=(to_file is not None))
        for row in zip(*(iter(words),) * nrows):
            for word in row:
                print(word.rjust(longest), end=" ", file=to_file, flush=(to_file is not None))
            print("", file=to_file, flush=(to_file is not None))
        print("", file=to_file, flush=(to_file is not None))


class Codenames:
    def __init__(self, cnt_rows=5, cnt_cols=5, cnt_agents=8, agg=.3):
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
        self.clue_cap = 3

        # Other
        self.vectors = np.array([])
        self.word_list = []
        self.weirdness = []
        self.word_to_index = {}
        self.codenames = []

    def load(self, datadir):
        # Glove word vectors
        print("...Loading vectors")
        # self.vectors = np.load("%s/glove.6B.300d.npy" % datadir)
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
        self.word_blobs = {w:TextBlob(w) for w in self.word_list}
        self.word_single = {w: self.word_blobs[w].words[0].singularize() 
                                if self.word_blobs[w].words 
                                else None 
                            for w in self.word_blobs.keys()}
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

    def generate_start_state(self):
        print("Starting the game...")
        words = random.sample(self.codenames, self.cnt_rows * self.cnt_cols)
        my_words = set(random.sample(words, self.cnt_agents))
        self.blacklist = set(my_words)
        ps = PorterStemmer()
        self.stems = [ps.stem(w) for w in words]

        self.negs = [w for w in words if w not in my_words]
        self.nm = (
            self.vectors @ np.array([self.word_to_vector(word) for word in self.negs]).T
        ).max(axis=1)

        self.clue_neg_sim = {}
        neg_vec = np.array([self.word_to_vector(word) for word in self.negs])

        for step, clue in enumerate(self.word_list):
            clue_vec = self.word_to_vector(clue)
            highest_neg_sim = float('-inf')
            for neg in neg_vec:
                neg_similarity = np.dot(clue_vec.T, neg)/(np.linalg.norm(clue_vec.T) * np.linalg.norm(neg))
                if neg_similarity > highest_neg_sim:
                    highest_neg_sim = neg_similarity
            self.clue_neg_sim[clue] = highest_neg_sim

        print("Spymaster ready!")
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
        self.clue_cap = 3

        print("New game start!", file=game_log, flush=True)
        reader.print_words(words, nrows=self.cnt_rows, to_file=game_log)

        while my_words:
            if turns_taken == 3:
                self.clue_cap = 2

            actions, costs = find_next_clue(tuple(words), tuple(my_words), self)
            clue, group = max(actions, key = lambda a: len(a[1]))
            # KAIS: uncomment line below to do kmeans instead
            #clue, group = find_next_clue_kmeans(tuple(words), tuple(my_words), self)
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
                'Clue: "{} {}" (remaining words {})'.format(
                    clue, len(group), len(my_words)
                )
            )
            print("Clue: %s, %s" % (clue, len(group)), file=game_log, flush=True)
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

        print("Game finished!", file=game_log, flush=True)
        print("Average clue count: %s" % str(sum(clue_count) / len(clue_count)), file=game_log, flush=True)
        print("Turns taken: %s" % turns_taken, file=game_log, flush=True)
        print("Wrong guesses: %s" % wrong_guesses, file=game_log, flush=True)

        score = 1.3602 * (sum(clue_count) / len(clue_count)) + 0.2524 * turns_taken + 0.4118 * wrong_guesses - 0.3789
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
