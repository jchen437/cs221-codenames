## Play Codenames with Word Embeddings

This repository implements a simple single-player version of the codenames game
by Vlaada Chvátil. The simulator is based on the Codenames simulator by [Thomas Ahle](https://github.com/thomasahle/codenames).

You can play as the agent or the spymaster, and an AI will take up the opposing role, using word embeddings to generate clues or guesses. This implementation only optimizes the AI spymaster for better clue generation; the AI agent implementation has not been modified from Ahle's simulator.

## Requirements

- Python 3
- Numpy
- NLTK
- TextBlob
- scikit-learn

## Getting Started
```
$ git clone git@github.com:jchen437/cs221-codenames.git
```

If you would like to use the full Wikipedia GloVe word embeddings, run the following. Otherwise, a set of embeddings trained on a subset of Wikipedia articles are already in the repository (see `dataset/wiki_subset_vectors.npy`).

```
$ sh get_glove.sh
```

To play a game:
```

$ python3 codenames.py
...Loading vectors
...Loading words
...Making word to index dict
...Loading codenames
Ready!

Will you be agent or spymaster?: agent
Starting the game...
Spymaster ready!
Thinking...

Clue: "metal 3" (remaining words 5)

   knife        mint        luck      carrot        pipe
   telescope    australia   beijing   dress         dwarf
   belt         dragon      ketchup   cast          ambulance
   fence        match       unicorn   millionaire   cook
   revolution   web         check     thief         arm

Guesses left: 3
Remaining words: 5
Your guess: knife
Correct!

```

## Logging Results

When playing games, the metrics used to create your score will be stored in a file `results.csv`. A more user-friendly display of the game, including the initial board, clues, and guesses taken will be stored in a file `game_log.txt`. As you play, additional game logs will be added to these files; previous games will not be overwritten.
