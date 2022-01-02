import os
import pickle

from matplotlib import pyplot as plt

DATA_PATH = "../data/"


def parse_pickle(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except pickle.UnpicklingError as err:
        print(f'Unable to unpickle "{path}" -- "{err}"; deleting pickle...')
        os.remove(path)
        return None, None, None
    except EOFError as err:
        print(f'Error in file "{path}" -- "{err}"; deleting pickle...')
        os.remove(path)
        return None, None, None

    winner = data[0]
    neighbours = data[1]
    states = data[2]

    return winner, neighbours, states


if __name__ == "__main__":
    winner_series = [[0] for _ in range(4)]
    game_lengths = []
    for filepath in [os.path.join(DATA_PATH, name) for name in os.listdir(DATA_PATH) if name.endswith(".pickle")]:
        winner, neighbours, states = parse_pickle(filepath)
        if winner is None:
            continue

        game_lengths.append(len(states))

        # print(filepath, winner, len(neighbours), len(states), len(states[0]))
        for i in range(4):
            winner_series[i].append(winner_series[i][-1] + winner[i])

    xs = list(range(len(winner_series[0])))
    for index, series in enumerate(winner_series):
        plt.plot(xs, series, label=f"Hráč na {index + 1}. pozici")

    plt.title("Závislost šance na výhru na pořadí hráčů")
    plt.xlabel("Počet celkem odehraných her")
    plt.ylabel("Počet výher")
    plt.grid(axis="both")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("../win_prob.pdf")

    plt.clf()
    plt.hist(game_lengths, density=False, bins="auto")
    plt.title("Histogram délky her")
    plt.ylabel("Počet her")
    plt.xlabel("Počet tahů")
    plt.tight_layout()
    # plt.show()
    plt.savefig("../game_len_hist.pdf")
