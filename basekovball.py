import csv
import numpy as np
import pandas as pd
import pathlib
import scipy.linalg as sp




base_runner_dist = [{}, 1, 2, 3, 12, 13, 23, 123]
# {} = bases empty &   23 = runners on 2nd & 3rd
# ({}, 2) = bases empty, 2 outs
# (123, 1) = bases loaded, 2 outs

state_space = [({}, 0), (1, 0), (2, 0), (3, 0), (12, 0), (13, 0), (23, 0), (123, 0),
               ({}, 1), (1, 1), (2, 1), (3, 1), (12, 1), (13, 1), (23, 1), (123, 1),
               ({}, 2), (1, 2), (2, 2), (3, 2), (12, 2), (13, 2), (23, 2), (123, 2),
               ]

# X blocks represent events where the number of outs do not change
# Y blocks represent events where the number of outs increase by 1 (but not end in 3rd out)
# Z blocks represent events where the number of outs increase by 2 (double plays) ** not accounted
#                                                                                   for in this model **

X0 = np.zeros([8,8])                    # events from 0 outs to 0 outs
Y0 = np.zeros([8,8])                    # events from 0 outs to 1 out
Z0 = np.zeros([8,8])                    # events from 0 outs to 2 outs ** not accounted for in this model **
X1 = np.zeros([8,8])                    # events from 1 outs to 2 outs
Y1 = np.zeros([8,8])                    # events from 1 outs to 2 outs
X2 = np.zeros([8,8])                    # events from 2 outs to 2 outs

O  = np.zeros([8,8])                    # impossible events (outs decreasing)

# H = transition matrix for a half-inning
H = np.block([
    [X0, Y0, Z0],
    [ O, X1, Y1],
    [ O,  O, X2]
])


def player_matrix(homeruns, triples, doubles, singles, walks, outs, plate_appearances):

    h = homeruns/plate_appearances
    t = triples/plate_appearances
    d = doubles/plate_appearances
    s = singles/plate_appearances
    w = walks/plate_appearances
    o = outs/plate_appearances


    # B = fundamental block matrix: represents possible transitions between states that do not result in an out
    B = np.array([
            [h, w+s,   d, t,       0,   0,   0,       0],
            [h,   0, d/2, t, w+(s/2), s/2, d/2,       0],
            [h, s/2,   d, t,       w, s/2,   0,       0],
            [h,   s,   d, t,       0,   w,   0,       0],
            [h,   0, d/2, t,     s/6, s/3, d/2, w+(s/2)],
            [h,   0, d/2, t,     s/2, s/2, d/2,       w],
            [h, s/2,   d, t,       0, s/2,   0,       w],
            [h,   0, d/2, t,     s/2, s/2, d/2,       w],
        ])

    # I = 8x8 outs identity matrix, transitions back to same inning and base runner dist. state
    #     with one more out, when out occurs that is not 3rd out

    I = np.zeros([8,8])
    np.fill_diagonal(I, o)

    # V = 8x1 outs column vector, transitions back to the next innings zero out no base runner state,
    #     after 3rd out
    V = np.full([8,1], o)


    # T = 217x217 transition matrix, for each player
    tb = sp.block_diag(B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, 0)
    v8 = np.block([V, np.zeros([8,7])])
    one = np.ones((1,1))
    v9 = np.concatenate((V, one), axis=0)

    offset = 8
    aux = np.empty((0, offset), int)
    ti = sp.block_diag(aux, I, I, v8, I, I, v8, I, I, v8, I, I, v8, I, I, v8, I, I, v8, I, I, v8, I, I, v8, I, I, v9)

    T = tb + ti

    return T


def game_matrix(file_name_input):
    f = open(file_name_input)
    csv_f = csv.DictReader(f)

    game_T_matrix = []

    for index, row in enumerate(csv_f):
        homeruns = int(row['homeruns'])
        triples = int(row['triples'])
        doubles = int(row['doubles'])
        singles = int(row['singles'])
        walks = int(row['walks'])
        outs = int(row['outs'])
        plate_appearances = int(row['plate_appearances'])

        player_T_matrix = player_matrix(homeruns, triples, doubles, singles, walks, outs, plate_appearances)

        game_T_matrix.append(player_T_matrix)

    return game_T_matrix


def run_value_matrix():

    # N = 8x8 runs matrix, the # of runs that score between all possible transitions that do not
    #     result in an out being recorded, for 1 half-inning of baseball

    N = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 1, 1, 0, 0, 0, 0],
        [2, 1, 1, 1, 0, 0, 0, 0],
        [2, 1, 1, 1, 0, 0, 0, 0],
        [3, 2, 2, 2, 1, 1, 1, 0],
        [3, 2, 2, 2, 1, 1, 1, 0],
        [3, 2, 2, 2, 1, 1, 1, 0],
        [4, 3, 3, 3, 2, 2, 2, 1]
    ])

    # R = 217x217 run value matrix, keeps track of # of runs scored throughout whole baseball game
    R = sp.block_diag(N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, 0)
    return R


def current_state():
    # C = Current state vector, keeps track of the probability that the game is in any given state,
    #     and used to determine when the game is over, C[0,217] > .99
    C = np.zeros([1,217])
    C[0,0] = 1

    return C

def play_ball(lineup, C, R, game_T_matrix):

    total_runs = 0
    hitter = 1

    while C[0,216] < 0.99:
        i = lineup[hitter - 1] - 1
        T = game_T_matrix[i]

        runs = np.dot(C, R*T)
        total_runs += np.sum(runs)

        C = np.dot(C, T)

        hitter = hitter + 1
        if hitter > 9:
            hitter = 1

    return total_runs

def clean_csv(file_name_input):
    assert (pathlib.Path(file_name_input).exists()), "ERROR:  Unable to find file in current working directory"
    return file_name_input



def clean_lineup(raw_lineup_input):
    default = "123456789"
    lineup = list(map(int, default))
    if len(raw_lineup_input) != 9:
        print("\nERROR: Batting order must contain 9 hitters (ex. 987654321) ...using default batting order instead")
        return lineup
    else:
        for i, no in enumerate(raw_lineup_input):
            try:
                lineup[i] = int(no)
            except:
                print("\n\nERROR: Batting order must contain only #'s (ex. 987654321) ...using default batting order instead")
                return lineup

    lineup = list(map(int, raw_lineup_input))

    return lineup


def lineup_card(file_name_input, lineup):
    df = pd.read_csv(file_name_input, index_col=0)

    lineup_card = []

    for index, i in enumerate(lineup, start=1):
        lineup_spot = i
        player_name = df.loc[(i, 'player_name')]
        position = df.loc[(i, 'position')]

        lineup_line = [(index, lineup_spot, player_name, position)]

        lineup_card += lineup_line

    lineup_card_df = pd.DataFrame(lineup_card, columns=['#', 'order', 'player_name', 'position'])

    print(lineup_card_df.to_string(index=False))

    return lineup_card_df

file_name_input = input("Please specify the file name containing player statistics? (ie. playerstats.csv ) \
                        \n\nRequirements: \
                        \n• File must be in working directory \
                        \n• File must be .csv with column and row headers, as follows: \
                        \n       order | player_name | position | homeruns | triples | doubles | singles | walks | outs | plate_appearances \
                        \n           1 | \
                        \n           2 | \
                        \n           3 | \
                        \n           4 | \
                        \n           5 | \
                        \n           6 | \
                        \n           7 | \
                        \n           8 | \
                        \n           9 | \n")

clean_csv(file_name_input)

raw_lineup_input = input("\nPlease specify a batting order which reorders the hitters \
                          \naccording to their original order in .csv player statistics file: \
                          \n     (ex. 123456789 = original order) \
                          \n     (ex. 987654321 = reverse order) \
                          \n     (ex. 123987456 = some other ordering) \n")


C = current_state()
R = run_value_matrix()

lineup = clean_lineup(raw_lineup_input)

print("\n", "\nThe following batting order...\n")

lineup_card = lineup_card(file_name_input, lineup)
game_T_matrix = game_matrix(file_name_input)

expected_runs = play_ball(lineup, C, R, game_T_matrix)

print("\n will produce ", expected_runs, "expected runs per game!", "\n")


