import csv
import math
from nltk.stem import PorterStemmer
import numpy as np
import os
import random
import re
from sklearn.cluster import KMeans
from textblob import TextBlob

import ngrams
import util

game_log = open("game_log.txt", "a+")
RESULTS_FILE_PATH = "results.csv"

clue_count = []
wrong_guesses = 0
turns_taken = 0

def log_result(score, clue_count, turns_taken, wrong_guesses):
    """ Logs scores and metrics for each game."""
    new_file = False
    if not os.path.isfile(RESULTS_FILE_PATH):
        new_file = True

    with open (RESULTS_FILE_PATH, 'a') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                'Final score',
                'Clue Count',
                'Sum Clue Count',
                'Turns Taken',
                'Wrong Guesses'
            ])
        result = [score, clue_count, sum(clue_count), turns_taken, wrong_guesses]
        writer.writerow(result)


# Readers to read in terminal input for guesses
class Reader:
    def read_picks(self, words, my_words, guesses_left):
        """
        Query the user for guesses.
        :param words: Words the user can choose from.
        :param my_words: Correct words.
        :param guesses_left: Number of guesses left
        :return: The words picked by the user.
        """
        raise NotImplementedError

    def read_clue(self, word_set):
        """
        Read a clue from the (spymaster) user.
        :param word_set: Valid words
        :return: The clue and number given.
        """
        raise NotImplementedError

    def print_words(self, words, nrows):
        """
        Prints a list of words as a 2d table, using `nrows` rows.
        :param words: Words to be printed.
        :param nrows: Number of rows to print.
        """
        raise NotImplementedError


class TerminalReader(Reader):
    def read_picks(self, words, my_words, guesses_left):
        """ Queries the user for guesses."""
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

    def read_clue(self, word_set):
        """ Reads a clue from the (spymaster) user."""
        while True:
            inp = input("Clue (e.g. 'car 2'): ").lower()
            match = re.match("(\w+)\s+(\d+)", inp)
            if match:
                clue, cnt = match.groups()
                if clue not in word_set:
                    print("I don't understand that word.")
                    continue
                return clue, int(cnt)

    def print_words(self, words, nrows, to_file=None):
        """ Prints words as a 2d board."""
        longest = max(map(len, words))
        print("", file=to_file, flush=(to_file is not None))
        for row in zip(*(iter(words),) * nrows):
            for word in row:
                print(word.rjust(longest),
                        end=" ",
                        file=to_file,
                        flush=(to_file is not None))
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
        self.word_to_index = {}
        self.codenames = []

    def load(self, datadir):
        """ Loads in emgedding vectors in words."""
        print("...Loading vectors")

        # self.vectors = np.load("%s/glove.6B.300d.npy" % datadir)
        self.vectors = np.load('%s/wiki_subset_vectors.npy' % datadir)

        # List of all embedding words
        print("...Loading words")
        
        # self.word_list = [w.lower().strip() for w in open("%s/glove_words.txt" % datadir)]
        self.word_list = [w.lower().strip() for w in open("%s/wiki_subset_words.txt" % datadir)]

        # Creates singularized version of each word for use in generating clues
        word_blobs = {w:TextBlob(w) for w in self.word_list}
        self.word_single = {w: word_blobs[w].words[0].singularize() 
                                if word_blobs[w].words 
                                else None 
                            for w in word_blobs.keys()}

        # Indexing back from word to indices
        print("...Making word to index dict")
        self.word_to_index = {w: i for i, w in enumerate(self.word_list)}

        # All words that are allowed to go onto the table
        print("...Loading codenames")
        self.codenames = [
            word for word in 
                (w.lower().strip().replace(" ", "-") for w in open("wordlist"))
            if word in self.word_to_index
        ]
        print("Ready!")

    def word_to_vector(self, word):
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

    def most_similar_to_given(self, clue, choices):
        """
        :param clue: Clue from the spymaster.
        :param choices: Choices on the table.
        :return: Which choice to go for.
        """
        clue_vector = self.word_to_vector(clue)
        return max(choices, key=lambda w: self.word_to_vector(w) @ clue_vector)

    def generate_start_state(self):
        """ Creates random board and selects word to be guessed, as
        well as initializing other vectors to be used in clue generation."""
        print("Starting the game...")

        words = random.sample(self.codenames, self.cnt_rows * self.cnt_cols)
        my_words = set(random.sample(words, self.cnt_agents))
        self.blacklist = set(my_words)

        # Generate stems of each word
        ps = PorterStemmer()
        self.stems = [ps.stem(w) for w in words]

        self.negs = [w for w in words if w not in my_words]
        self.nm = (
            self.vectors @ np.array([self.word_to_vector(word) for word in self.negs]).T
        ).max(axis=1)

        # To be used in the k-means model
        self.clue_neg_sim = {}
        neg_vec = np.array([self.word_to_vector(word) for word in self.negs])

        for step, clue in enumerate(self.word_list):
            clue_vec = self.word_to_vector(clue)
            highest_neg_sim = float('-inf')
            for neg in neg_vec:
                neg_similarity = np.dot(clue_vec.T, neg) / (np.linalg.norm(clue_vec.T) * np.linalg.norm(neg))
                if neg_similarity > highest_neg_sim:
                    highest_neg_sim = neg_similarity
            self.clue_neg_sim[clue] = highest_neg_sim

        print("Spymaster ready!")
        return words, my_words

    def play_spymaster(self, reader):
        """
        Play a complete game, with the robot being the spymaster.
        """
        # Metrics for calculating score
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

            # To find clues with K-means instead of search, uncomment the line below
            # clue, group = find_next_clue_kmeans(tuple(words), tuple(my_words), self)
            
            group_scores = np.array(
                [self.word_to_vector(w) for w in group]
            ) @ self.word_to_vector(clue)

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
            print()

            guesses_left = len(group)
            while guesses_left:
                reader.print_words(words, nrows=self.cnt_rows)
                pick = reader.read_picks(words, my_words, guesses_left)
                if pick not in words:
                    print("That isn't an option. Please try again!")
                    continue
                words[words.index(pick)] = "---"
                if pick in my_words:
                    my_words.remove(pick)
                    guesses_left -= 1
                else:
                    guesses_left = 0

        print("Game finished!", file=game_log, flush=True)
        print("Average clue count: %s" % str(sum(clue_count) / len(clue_count)), file=game_log, flush=True)
        print("Turns taken: %s" % turns_taken, file=game_log, flush=True)
        print("Wrong guesses: %s" % wrong_guesses, file=game_log, flush=True)

        score = 1.3602 * (sum(clue_count) / len(clue_count)) + 0.2524 * turns_taken + 0.4118 * wrong_guesses - 0.3789
        print("final score:", score)

        log_result(score, clue_count, turns_taken, wrong_guesses)

    def play_agent(self, reader):
        """
        Play a complete game, with the AI being the agent.
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


class CodenamesSearchProblem(util.SearchProblem):
    """ A search model used to generate clues for the AI spymaster."""
    def __init__(self, board, my_words, game):
        self.board = board
        self.my_words = my_words
        self.game = game # instance of CodeNames class

    def startState(self):
        return (self.board, self.my_words)

    def isEnd(self, state):
        board, my_words = state
        return not board or not my_words

    def succAndCost(self, state):
        """ Return a list of (action, newState, cost) tuples corresponding to 
        edges coming out of |state|."""
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
            # Ignore clues that are too obscure
            prob = ngrams.Pwords([clue])
            if prob < 1e-10:
                continue

            # Stem and singularize potential clues
            ps = PorterStemmer()
            single = self.game.word_single[clue]
            if not single:
                continue
            stem = ps.stem(single)

            # Ignore clues that are too similar to other words on the board
            if (max(scores) <= lower_bound or stem in self.game.stems 
                or clue in self.game.blacklist or stem in self.game.blacklist):
                continue

            ss = sorted((s, i) for i, s in enumerate(scores))[-self.game.clue_cap:]
            real_score, j = max(
                ((s - lower_bound) * ((len(ss) - j) ** self.game.agg - .99), j)
                for j, (s, _) in enumerate(ss)
            )
            group = [my_words[i] for _, i in ss[j:]]
            clue_groups.append((real_score, (clue, group)))
        return [p[1] for p in sorted(clue_groups)[-1000:]]

    def cost(self, clue, group, board, my_words):
        """ Cost function for the search problem. Calculates the
        negative sum of cosine similarites between the clue and 
        the input |my_words|."""
        negs = self.game.negs
        clue_vec = self.game.word_to_vector(clue)
        cost = 0
        for word in group:
            word_vec = self.game.word_to_vector(word)
            cost -= np.dot(word_vec.T, clue_vec)/(np.linalg.norm(word_vec.T) * np.linalg.norm(clue_vec))
        return cost


def print_to_txt(file, string):
    with open(file, "a") as f:
        f.write(string)


def find_next_clue(board, my_words, game):
    """ Use the search problem to generate clues for the AI spymaster."""
    ucs = util.UniformCostSearch(verbose=0)
    print("Thinking", end="", flush=True)
    ucs.solve(CodenamesSearchProblem(board, my_words, game))
    return ucs.actions, ucs.totalCost


def find_next_clue_kmeans(board, my_words, game):
    """ Generate clues for the AI spymaster using K-means."""
    pos_vec = np.array([game.word_to_vector(word) for word in my_words])
    kmeans = None
    K = 4
    if len(my_words) > K:
        kmeans = KMeans(n_clusters=K, random_state=0).fit(pos_vec)
    else:
        scaled_K = math.ceil(len(my_words) / 2)
        kmeans = KMeans(n_clusters=scaled_K, random_state=0).fit(pos_vec)
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
        # Ignore clues that are too obscure
        prob = ngrams.Pwords([clue])
        if prob < 1e-10:
            continue

        # Stem and singularize potential clues
        ps = PorterStemmer()
        single = game.word_single[clue]
        if not single:
            continue
        stem = ps.stem(single)
        if stem in game.stems or clue in game.blacklist or stem in game.blacklist:
            continue

        clue_vec = game.word_to_vector(clue)
        highest_neg_sim = game.clue_neg_sim[clue]
        highest = float('-inf')
        highest_center_index = 0
        highest_word = None

        # loop over all centers, find the cluster which is closest to the clue
        for i, center in enumerate(centers):
            # Clue's closest cluster should have more words
            # Clue should be close to center of the cluster
            # Clue should be far from most similar negative word
            similarity = np.dot(clue_vec.T, center)/(np.linalg.norm(clue_vec.T) * np.linalg.norm(center))
            weight = similarity * cluster_counts[i] - 0.1 * highest_neg_sim 
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
