from reasoning_core.template import Problem, Task, edict, Config
from reasoning_core.utils import score_scalar
import random



class nimConfig(Config):
    min_nb_piles: int = 3
    max_nb_piles: int = 5
    min_piles_size: int = 5
    max_piles_size: int = 7


    def update(self,c):
        self.min_nb_piles += c
        self.max_nb_piles += c
        self.min_piles_size += c
        self.max_piles_size += c

class Nim(Task):

    def __init__(self, config=nimConfig()):
        super().__init__(config=config)


    def generate(self):
        nb_piles = random.randint(self.config.min_nb_piles, self.config.max_nb_piles)
        pile = []
        for _ in range(nb_piles):
            list_num = random.randint(self.config.min_piles_size, self.config.max_piles_size)
            pile.append(list_num)

        nim_sum = 0
        for num in pile:
            nim_sum ^= num
        answer = "Oui" if nim_sum > 0 else "Non"
        metadata = edict({
            "pile": pile,
            "nim_sum": nim_sum
        })
        metadata.cot = self.get_cot(metadata)
        return Problem(metadata=metadata, answer=answer)

    def prompt(self,metadata):
        piles_str = ", ".join(str(p) for p in metadata.pile)
        return (f"Voici une partie du jeu de Nim en cours. "
                f"Les tas contiennent le nombre d'allumettes suivant : {piles_str}. "
                f"C'est à ton tour de jouer. En supposant que les deux joueurs jouent de manière optimale, "
                f"peux-tu forcer la victoire ? Réponds uniquement par 'Oui' ou 'Non'.")

    def score_answer(self, answer, entry):
        return score_scalar(answer, entry)

    def get_cot(self, metadata):
        nim_sum = 0
        steps = [f"Tas initiaux : {metadata.pile}"]
        for num in metadata.pile:
            steps.append(f"{nim_sum} ^ {num} = {nim_sum ^ num}")
            nim_sum ^= num
        steps.append(f"Nim-sum final = {nim_sum}")
        steps.append("Gagnant : " + ("Oui" if nim_sum > 0 else "Non"))
        return "\n".join(steps)