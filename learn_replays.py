from pickle import load

if __name__ == "__main__":
    with open(r"/home/gilsson/StarTrain/StarTrainSaved/stats_replay/stats.pkl", "rb") as file:
        replay = load(file)

    print(replay["actions"])
