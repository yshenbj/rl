from normal_form_game import NormalFormGame


prisoners_dilemma = NormalFormGame(
    "Prisoner A",
    "Prisoner B",
    ["admit", "deny"],
    ["admit", "deny"],
    {
        ("admit", "admit"): (-2, -2),
        ("admit", "deny"): (0, -4),
        ("deny", "admit"): (-4, 0),
        ("deny", "deny"): (-1, -1),
    },
)
prisoners_dilemma.visualise()

split_or_steal = NormalFormGame(
    "Agent 1",
    "Agent 2",
    ["split", "steal"],
    ["split", "steal"],
    {
        ("split", "split"): (1, 1),
        ("split", "steal"): (0, 2),
        ("steal", "split"): (2, 0),
        ("steal", "steal"): (0, 0),
    },
)
split_or_steal.visualise()

matching_pennies = NormalFormGame(
    "Player Even",
    "Player Odd",
    ["heads", "tails"],
    ["heads", "tails"],
    {
        ("heads", "heads"): (1, -1),
        ("heads", "tails"): (-1, 1),
        ("tails", "heads"): (-1, 1),
        ("tails", "tails"): (1, -1),
    },
)
matching_pennies.visualise()

security_game = NormalFormGame(
    "Defender",
    "Adversary",
    ["T1", "T2"],
    ["T1", "T2"],
    {
        ("T1", "T2"): (5, -3),
        ("T1", "T1"): (-1, 1),
        ("T2", "T1"): (-5, 5),
        ("T2", "T2"): (2, -1),
    },
)
security_game.visualise()

