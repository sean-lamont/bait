import glob
import pickle
import sys
from multiprocessing import Pool

from lean_dojo.interaction.dojo import Dojo
from lean_dojo.interaction.dojo import TacticState, ProofFinished
from loguru import logger
from tqdm import tqdm

from experiments.end_to_end.common import remove_marks


# todo extend to other environments, currently only works for LeanDojo

def verify_proof(trace):
    proof = trace.proof
    thm = trace.theorem
    dojo, init_state = Dojo(thm, timeout=6000).__enter__()

    state = init_state
    logger.info(f'Verifying proof of {thm.full_name}')

    for tactic in proof:
        tactic_ = remove_marks(tactic)
        logger.info(f'Running tactic {tactic_} to verify \n{state.pp}\n')
        response = dojo.run_tac(state, tactic_)
        if isinstance(response, TacticState):
            state = response
        elif isinstance(response, ProofFinished):
            return True
        else:
            logger.warning(f'Response {response} to tactic {tactic_} is not a TacticState or ProofFinished')
            return False


def check_file(trace):
    res = verify_proof(trace)
    if not res:
        logger.warning(f'Proof of {trace.theorem.full_name} is invalid')
        return False
    else:
        logger.info(f'Proof of {trace.theorem.full_name} is valid')
        return True


def process_file(file):
    try:
        trace = pickle.load(open(file, 'rb'))
    except:
        logger.warning(f'Error loading {file}')
        return 0, 0
    verified_proof = 0

    if not trace.proof:
        logger.info(f'No proof for {trace.theorem.full_name}')
        found_proof = 0
    else:
        found_proof = 1
        try:
            res = check_file(trace)
            if res:
                verified_proof = 1
        except Exception as e:
            logger.warning(f'Error verifying proof of {file}: {e}')

    return found_proof, verified_proof


if __name__ == '__main__':
    # get trace_dir from system arguments
    trace_dir = sys.argv[1]
    num_procs = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # files = list(glob.glob(trace_dir + '/*'))
    files = list(glob.glob(trace_dir))

    # total_proofs = 0
    # verified_proofs = 0

    # for file in tqdm(files):
    #     trace = pickle.load(open(file, 'rb'))
    #     if not trace.proof:
    #         logger.info(f'No proof for {trace.theorem.full_name}')
    #     else:
    #         total_proofs += 1
    #         try:
    #             res = check_file(trace)
    #             if res:
    #                 verified_proofs += 1
    #         except Exception as e:
    #             logger.warning(f'Error verifying proof of {file}: {e}')
    #             continue

    # multithread the above instead:
    #
    with Pool(num_procs) as p:
        results = list(tqdm(p.imap(process_file, files), total=len(files)))
        total_proofs = sum([r[0] for r in results])
        verified_proofs = sum([r[1] for r in results])

    logger.info(f'Valid proofs: {verified_proofs}/{total_proofs}')
