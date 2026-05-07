from reasoning_core.template import Problem, Task, edict, Config
from dataclasses import dataclass
import random
import re


MOVES = ("rock", "paper", "scissors")

BEATS = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


def _rules_to_text():
    return (
        "- rock beats scissors\n"
        "- scissors beats paper\n"
        "- paper beats rock\n"
        "- if both players choose the same move, the result is Draw"
    )


def _round_winner(move_a, move_b, player_a="Alice", player_b="Bob"):
    if move_a == move_b:
        return "Draw"

    if BEATS[move_a] == move_b:
        return player_a

    return player_b


def _normalize_winner_label(answer):
    cleaned = re.sub(r"[^a-zA-Z]", "", str(answer).strip()).lower()

    labels = {
        "alice": "Alice",
        "bob": "Bob",
        "draw": "Draw",
        "tie": "Draw",
    }

    return labels.get(cleaned, "")


def _score_winner_answer(answer, entry):
    prediction = _normalize_winner_label(answer)
    reference = _normalize_winner_label(entry.answer)
    return 1 if prediction == reference else 0


def _round_cot(move_a, move_b, winner, player_a="Alice", player_b="Bob"):
    if winner == "Draw":
        return f"Both players choose {move_a}, so the result is Draw."

    if winner == player_a:
        return f"{move_a} beats {move_b}, so {player_a} wins."

    return f"{move_b} beats {move_a}, so {player_b} wins."


@dataclass
class RockPaperScissorsOutcomeConfig(Config):
    difficulty: int = 0

    def update(self, c):
        self.difficulty += c


class RockPaperScissorsOutcome(Task):
    """
    Simple task:
    given two moves, determine the winner of one rock-paper-scissors round.
    """

    def __init__(self, config=RockPaperScissorsOutcomeConfig()):
        super().__init__(config=config)

    def generate(self):
        player_1 = "Alice"
        player_2 = "Bob"

        player_1_move = random.choice(MOVES)
        player_2_move = random.choice(MOVES)

        winner = _round_winner(
            player_1_move,
            player_2_move,
            player_a=player_1,
            player_b=player_2,
        )

        meta = edict(
            player_1=player_1,
            player_2=player_2,
            player_1_move=player_1_move,
            player_2_move=player_2_move,
            rules=_rules_to_text(),
            cot=_round_cot(
                player_1_move,
                player_2_move,
                winner,
                player_a=player_1,
                player_b=player_2,
            ),
        )

        return Problem(metadata=meta, answer=winner)

    def prompt(self, metadata):
        return (
            "Rock-paper-scissors round.\n\n"
            "Rules:\n"
            f"{metadata.rules}\n\n"
            f"{metadata.player_1} plays {metadata.player_1_move}.\n"
            f"{metadata.player_2} plays {metadata.player_2_move}.\n\n"
            "Who wins this round?\n"
            "The answer is exactly one of: Alice, Bob, Draw."
        )

    def score_answer(self, answer, entry):
        return _score_winner_answer(answer, entry)


@dataclass
class RockPaperScissorsTournamentConfig(Config):
    min_rounds: int = 3
    max_rounds: int = 5

    def update(self, c):
        self.min_rounds += c
        self.max_rounds += c


class RockPaperScissorsTournament(Task):
    """
    Improved task:
    given several rounds, determine the winner of the tournament.
    """

    def __init__(self, config=RockPaperScissorsTournamentConfig()):
        super().__init__(config=config)

    def generate(self):
        player_1 = "Alice"
        player_2 = "Bob"

        min_rounds = max(1, self.config.min_rounds)
        max_rounds = max(min_rounds, self.config.max_rounds)

        n_rounds = random.randint(min_rounds, max_rounds)

        score_player_1 = 0
        score_player_2 = 0
        rounds = []
        cot_lines = []

        for i in range(1, n_rounds + 1):
            player_1_move = random.choice(MOVES)
            player_2_move = random.choice(MOVES)

            winner = _round_winner(
                player_1_move,
                player_2_move,
                player_a=player_1,
                player_b=player_2,
            )

            if winner == player_1:
                score_player_1 += 1
            elif winner == player_2:
                score_player_2 += 1

            rounds.append(
                {
                    "index": i,
                    "player_1_move": player_1_move,
                    "player_2_move": player_2_move,
                    "winner": winner,
                }
            )

            cot_lines.append(
                f"Round {i}: "
                + _round_cot(
                    player_1_move,
                    player_2_move,
                    winner,
                    player_a=player_1,
                    player_b=player_2,
                )
            )

        if score_player_1 > score_player_2:
            tournament_winner = player_1
        elif score_player_2 > score_player_1:
            tournament_winner = player_2
        else:
            tournament_winner = "Draw"

        cot_lines.append(
            f"Final score: {player_1} {score_player_1}, {player_2} {score_player_2}."
        )
        cot_lines.append(f"Tournament result: {tournament_winner}.")

        meta = edict(
            player_1=player_1,
            player_2=player_2,
            rounds=rounds,
            score_player_1=score_player_1,
            score_player_2=score_player_2,
            rules=_rules_to_text(),
            cot="\n".join(cot_lines),
        )

        return Problem(metadata=meta, answer=tournament_winner)

    def prompt(self, metadata):
        rounds_text = "\n".join(
            f"Round {r['index']}: "
            f"{metadata.player_1} plays {r['player_1_move']}. "
            f"{metadata.player_2} plays {r['player_2_move']}."
            for r in metadata.rounds
        )

        return (
            "Rock-paper-scissors tournament.\n\n"
            "Rules:\n"
            f"{metadata.rules}\n\n"
            f"{rounds_text}\n\n"
            "Each round winner gets 1 point. Draw rounds give no point.\n"
            "Who wins the tournament?\n"
            "The answer is exactly one of: Alice, Bob, Draw."
        )

    def score_answer(self, answer, entry):
        return _score_winner_answer(answer, entry)
