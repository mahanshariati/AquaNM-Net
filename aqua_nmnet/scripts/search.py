import functools
import os
import random
import time

import numpy as np
import torch

from aqua_nmnet.config import (
    SEARCH_INPUT_DIR,
    SEARCH_TARGET_DIR,
)
from aqua_nmnet.models.model import AquaNMNet
from aqua_nmnet.evaluation.tester_model import get_cand_err


print = functools.partial(print, flush=True)
torch.backends.cudnn.deterministic = False

device_id = 0
PSNR_TIE_DELTA = 0.07

# --------- paths & settings ---------
CKPT_FILE = ""

args = {
    "choice": 9,
    "layers": 13,
    "base_channels": 48,
    "log_dir": "log",
    "max_epochs": 50,
    "select_num": 6,
    "population_num": 30,
    "top_k": 10,
    "m_prob": 0.1,
    "crossover_num": 10,
    "mutation_num": 10,
    "image_size": 256,
    "image_path": SEARCH_INPUT_DIR,
    "gt_path":    SEARCH_TARGET_DIR,
}

def random_choice(items):
    items = tuple(items)
    return items[
        np.random.randint(len(items))
    ]

class EvolutionSearcher(object):
    def __init__(self):
        self.args = args

        self.max_epochs = args["max_epochs"]
        self.select_num = args["select_num"]
        self.top_k = args["top_k"]
        self.population_num = args["population_num"]
        self.m_prob = args["m_prob"]
        self.crossover_num = args["crossover_num"]
        self.mutation_num = args["mutation_num"]

        # load model weights
        self.model = AquaNMNet(args["base_channels"])
        checkpoint = torch.load(
            CKPT_FILE,
            map_location="cpu",
        )
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint
        
        self.model.load_state_dict(
            state,
            strict=True
        )

        self.log_dir = args["log_dir"]
        self.checkpoint_path = os.path.join(
            self.log_dir,
            "checkpoint.pth.tar",
        )
        
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], self.top_k: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = args["layers"]
        self.nr_state = args["choice"]


    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        info = {
            "memory": self.memory,
            "candidates": self.candidates,
            "vis_dict": self.vis_dict,
            "keep_top_k": self.keep_top_k,
            "epoch": self.epoch,
        }
        
        torch.save(info, self.checkpoint_path)
        print("save checkpoint to", self.checkpoint_path)

        # also save best architecture to a txt file
        best = self.keep_top_k[self.top_k][0]
        best_err = self.vis_dict[best]["err"]
        best_path = os.path.join(self.log_dir, "best_arch.txt")
        with open(best_path, "w") as f:
            f.write("BEST_ARCH {}\n".format(list(best)))
            f.write("METRICS (PSNR, SSIM) {}\n".format(best_err))
        print("Best architecture saved to", best_path)
        print("BEST_ARCH =", list(best), "ERR =", best_err)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print("No previous search checkpoint found.")
            return False
    
        info = torch.load(
            self.checkpoint_path,
            map_location="cpu",
        )
    
        self.memory = info["memory"]
        self.candidates = info["candidates"]
        self.vis_dict = info["vis_dict"]
        self.keep_top_k = info["keep_top_k"]
        self.epoch = info["epoch"]

        top_candidates = self.keep_top_k[
            self.top_k
        ]
    
        print("Resumed search")
        print("Starting epoch:", self.epoch)
        print("Evaluated architectures:", len(self.vis_dict))
        if top_candidates:
            best = top_candidates[0]
            print(
                "Current best:",
                best,
                self.vis_dict[best]["err"],
            )
    
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if "visited" in info:
            return False

        print("Evaluating cand:", cand)
        info["err"] = get_cand_err(self.model, cand, self.args)  # (psnr, ssim)
        info["visited"] = True
        return True

    def rank_candidates(self, candidates):
        
        candidates = list(
            dict.fromkeys(candidates)
        )
    
        candidates = sorted(
            candidates,
            key=lambda cand: self.vis_dict[cand]["err"][0],
            reverse=True
        )
    
        ranked = []
        idx = 0
    
        while idx < len(candidates):
            group_best_psnr = self.vis_dict[
                candidates[idx]
            ]["err"][0]
    
            group = []
    
            while idx < len(candidates):
                cand = candidates[idx]
    
                cand_psnr = self.vis_dict[
                    cand
                ]["err"][0]
    
                if (
                    group_best_psnr - cand_psnr
                    >= PSNR_TIE_DELTA
                ):
                    break
    
                group.append(cand)
                idx += 1
    
            group.sort(
                key=lambda cand: (
                    self.vis_dict[cand]["err"][1],
                    self.vis_dict[cand]["err"][0]
                ),
                reverse=True
            )
    
            ranked.extend(group)
    
        return ranked
    
    
    def update_top_k(self, candidates, *, k):
        assert k in self.keep_top_k
    
        print("select ......")
    
        combined = (
            self.keep_top_k[k]
            + list(candidates)
        )
    
        ranked = self.rank_candidates(
            combined
        )
    
        self.keep_top_k[k] = ranked[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
            for cand in cands:
                yield cand

    def get_random(self, num):
        print("random select ........")
        cand_iter = self.stack_random_cand(
            lambda: tuple(np.random.randint(self.nr_state) for _ in range(self.nr_layer))
        )
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print("random {}/{}".format(len(self.candidates), num))
        print("random_num = {}".format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print("mutation ......")
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = list(
                random_choice(
                    self.keep_top_k[k]
                )
            )
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print("mutation {}/{}".format(len(res), mutation_num))

        print("mutation_num = {}".format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print("crossover ......")
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random_choice(
                self.keep_top_k[k]
            )
            
            p2 = random_choice(
                self.keep_top_k[k]
            )
            
            return tuple(
                random_choice((i, j))
                for i, j in zip(p1, p2)
            )
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print("crossover {}/{}".format(len(res), crossover_num))

        print("crossover_num = {}".format(len(res)))
        return res


    def search(self):
        print(
            "population_num = {} select_num = {} mutation_num = {} "
            "crossover_num = {} max_epochs = {}".format(
                self.population_num,
                self.select_num,
                self.mutation_num,
                self.crossover_num,
                self.max_epochs,
            )
        )
    
        self.load_checkpoint()
    
    
        if len(self.candidates) < self.population_num:
            self.get_random(self.population_num)
    
        while self.epoch < self.max_epochs:
            print(
                "epoch = {}".format(
                    self.epoch
                )
            )
    
            self.memory.append(
                list(self.candidates)
            )
    
            self.update_top_k(
                self.candidates,
                k=self.select_num
            )
            
            self.update_top_k(
                self.candidates,
                k=self.top_k
            )
    
            print(
                "epoch = {} : top {} result".format(
                    self.epoch,
                    len(self.keep_top_k[self.top_k])
                )
            )
    
            for i, cand in enumerate(
                self.keep_top_k[self.top_k]
            ):
                print(
                    "No.{} {}  ERR(PSNR,SSIM) = {}".format(
                        i + 1,
                        cand,
                        self.vis_dict[cand]["err"]
                    )
                )
    
                print(
                    "ops:",
                    list(cand)
                )
    

            if self.epoch + 1 >= self.max_epochs:
                self.epoch += 1
                break
    
            mutation = self.get_mutation(
                self.select_num,
                self.mutation_num,
                self.m_prob
            )
    
            crossover = self.get_crossover(
                self.select_num,
                self.crossover_num
            )
    
            self.candidates = (
                mutation + crossover
            )

            
            self.get_random(
                self.population_num
            )
    
            self.epoch += 1

            self.save_checkpoint()
    
        self.save_checkpoint()


def main():
    seed = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(device_id)

    if not os.path.isdir(
        args["image_path"]
    ):
        raise FileNotFoundError(
            "Search input directory not found: "
            f"{args['image_path']}"
        )

    if not os.path.isdir(
        args["gt_path"]
    ):
        raise FileNotFoundError(
            "Search GT directory not found: "
            f"{args['gt_path']}"
        )

    if not CKPT_FILE:
        raise ValueError(
            "CKPT_FILE must point to a trained AquaNMNet checkpoint."
        )
    
    if not os.path.isfile(CKPT_FILE):
        raise FileNotFoundError(
            f"Model checkpoint not found: {CKPT_FILE}"
        )

    start_time = time.time()

    searcher = EvolutionSearcher()
    searcher.search()

    print(
        "total searching time = "
        "{:.2f} hours".format(
            (time.time() - start_time)
            / 3600.0
        )
    )

if __name__ == "__main__":
    main()
