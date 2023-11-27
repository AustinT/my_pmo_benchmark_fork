from __future__ import annotations
import heapq
import logging
from pprint import pformat
import random
import sys


import numpy as np
import joblib
from rdkit import rdBase, Chem
from rdkit.Chem import rdFingerprintGenerator
rdBase.DisableLog('rdApp.error')

import gpytorch
import torch
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound

from trf23.tanimoto_gp import TanimotoKernelGP
from mol_ga import default_ga


from main.optimizer import BaseOptimizer

FP_DIM = 2048
FP_RADIUS = 3
GP_MEAN = 0.00

stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
stream_handler.setFormatter(formatter)

acq_opt_logger = logging.getLogger("acq_opt_logger")
bo_loop_logger = logging.getLogger("bo_loop_logger")
bo_loop_logger.setLevel(logging.DEBUG)
bo_loop_logger.addHandler(stream_handler)


def smiles_to_fingerprint_arr(smiles_list: list[str],) -> np.array:
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS,fpSize=FP_DIM)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [mfpgen.GetCountFingerprintAsNumPy(m) for m in mols]
    return np.asarray(fps, dtype=float)


def eval_acq_function_on_smiles(
    smiles_list: list[str],
    acquisition_function: AcquisitionFunction,
    device: torch.device,
    screen_batch_size: int = 1000,
) -> list[float]:
    """
    Evaluate an acquisition function on a list of smiles.
    
    Return list of acquisition function values.
    """
    acq_fn_values = []
    for batch_start in range(0, len(smiles_list), screen_batch_size):
        screen_batch = smiles_list[batch_start:batch_start + screen_batch_size]
        screen_batch_fp = smiles_to_fingerprint_arr(screen_batch,)
        with torch.no_grad():
            screen_batch_acq = acquisition_function(torch.as_tensor(screen_batch_fp).unsqueeze(1).to(device))
            acq_fn_values.extend(screen_batch_acq.cpu().numpy().tolist())
    return acq_fn_values


class TanimotoGPBO_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "tanimoto_gpbo"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)
        bo_loop_logger.info(f"Running with config: {pformat(config)}")

        # Canonicalize all smiles and remove duplicates
        # (otherwise can could potentially cause bugs)
        bo_loop_logger.info("Canonicalizing all smiles")
        self.all_smiles = list(set([Chem.CanonSmiles(s) for s in self.all_smiles]))

        # Randomly choose initial smiles
        starting_population = np.random.choice(self.all_smiles, config["num_start_samples"]).tolist()
        starting_population_scores = self.oracle(starting_population)
        known_smiles_scores = {s: score for s, score in zip(starting_population, starting_population_scores)}

        # Run BO loop
        ucb_beta_arr = np.logspace(0, -2, 10)
        bo_iter = 0
        rng = random.Random(self.seed)
        while not self.finish and bo_iter < config["max_bo_iter"]:
            bo_iter += 1
            bo_loop_logger.info(f"Start BO iteration {bo_iter}")

            # Featurize known smiles
            smiles_train = list(known_smiles_scores.keys())
            scores_train = np.asarray([known_smiles_scores[s] for s in smiles_train])
            fp_train = smiles_to_fingerprint_arr(smiles_train,)

            # Make GP and set hyperparameters,
            # also turning off gradients because we won't fit the model
            torch.set_default_dtype(torch.float64)  # higher precision for GP
            gp_model = TanimotoKernelGP(
                train_x=torch.as_tensor(fp_train),
                train_y=torch.as_tensor(scores_train),
                kernel="T_MM",
                mean_obj=gpytorch.means.ConstantMean(),
            )
            gp_model.covar_module.raw_outputscale.requires_grad_(False)
            gp_model.mean_module.constant.requires_grad_(False)
            gp_model.likelihood.raw_noise.requires_grad_(False)
            gp_model.mean_module.constant.data.fill_(GP_MEAN)
            gp_model.covar_module.outputscale = 1.0
            gp_model.likelihood.noise = 1e-4  # little (but non-zero) noise, mostly for model misspecification
            gp_model.eval()

            # Potentially move to GPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                gp_model = gp_model.cuda()
            else:
                device = torch.device("cpu")

            # Define acqusition function
            # We use UCB with cyclic beta,
            # chosen to be between 0.01 (pure exploitation)
            # and 1.0 (pure exploration, i.e. random points will have a higher value than incumbant best point)
            ucb_beta = ucb_beta_arr[(bo_iter - 1) % len(ucb_beta_arr)]
            bo_loop_logger.info(f"UCB beta: {ucb_beta:.3f}")
            acq_function = UpperConfidenceBound(gp_model, beta=ucb_beta) 

            # Pick starting population for acq opt GA
            ga_start_smiles = (
                rng.choices(self.all_smiles, k=config["ga_start_population_size"]) + 
                list(known_smiles_scores.keys())
            )
            ga_start_smiles = list(set(ga_start_smiles))  # remove duplicates
            
            # Optimize acquisition function
            bo_loop_logger.debug("Starting acquisition function optimization")
            def _ga_inner_function(smiles_list):
                with torch.no_grad():
                    out = eval_acq_function_on_smiles(smiles_list, acq_function, device)
                return out
            with joblib.Parallel(n_jobs=4) as parallel:
                acq_opt_output = default_ga(
                    starting_population_smiles=ga_start_smiles,
                    scoring_function=_ga_inner_function,
                    max_generations=config["ga_max_generations"],
                    offspring_size=config["ga_offspring_size"],
                    population_size=config["ga_population_size"],
                    rng=rng,
                    parallel=parallel,
                )
            del _ga_inner_function  # might cause excess GPU memory to be used
            top_ga_smiles = sorted(acq_opt_output.scoring_func_evals.items(), key=lambda x: x[1], reverse=True)
            batch_candidate_smiles = [s for s, _ in top_ga_smiles]

            # Choose a batch of the top SMILES to evaluate which have not been measured before and log their acquisition function values
            eval_batch = [s for s in batch_candidate_smiles if s not in known_smiles_scores][:config["bo_batch_size"]]
            eval_batch_acq_values = eval_acq_function_on_smiles(eval_batch, acq_function, device)
            bo_loop_logger.debug(f"Eval batch SMILES: {pformat(eval_batch)}")
            bo_loop_logger.debug(f"Eval batch acq values: {pformat(eval_batch_acq_values)}")
            
            # Score the batch with the oracle
            eval_batch_scores = self.oracle(eval_batch)
            bo_loop_logger.debug(f"Eval batch scores: {eval_batch_scores}")
            known_smiles_scores.update({s: score for s, score in zip(eval_batch, eval_batch_scores)})
            
            # Final message
            bo_loop_logger.info(f"End BO iteration {bo_iter}. Top scores so far:\n{pformat(heapq.nlargest(5, known_smiles_scores.values()))}")

            # Free up GPU memory for next iteration by deleting the model
            del acq_function, gp_model
            torch.cuda.empty_cache()
        
        if not self.finish:
            bo_loop_logger.info(
                f"Budget not used even after {config['max_bo_iter']} BO iterations."
                " Will now choose random SMILES to fill the budget."
            )
            all_smiles_copy = list(self.all_smiles)
            random.shuffle(all_smiles_copy)
            for s in all_smiles_copy:
                self.oracle([s])
                if self.finish:
                    break
        
        bo_loop_logger.info(f"Finished BO loop.")
