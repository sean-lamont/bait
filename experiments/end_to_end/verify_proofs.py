import glob
import pickle
import sys

from lean_dojo.interaction.dojo import Dojo
from lean_dojo.interaction.dojo import TacticState, ProofFinished
from loguru import logger
from tqdm import tqdm

from experiments.end_to_end.common import remove_marks


# todo check trace for environment, and handle separately
# todo multiprocessing

def verify_proof(trace):
    proof = trace.proof
    thm = trace.theorem
    dojo, init_state = Dojo(thm, hard_timeout=6000).__enter__()

    state = init_state
    logger.info(f'Verifying proof of {thm.full_name}')

    for tactic in proof:
        tactic_ = remove_marks(tactic)
        logger.info(f'Running tactic {tactic_} to verify \n{state.pp}\n')
        response = dojo.run_tac(state, tactic_)
        if isinstance(response, TacticState):
            state = response
        elif isinstance(response, ProofFinished):
            dojo._cleanup()
            return True
        else:
            dojo._cleanup()
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


if __name__ == '__main__':
    # get trace_dir from system arguments
    trace_dir = sys.argv[1]

    files = list(glob.glob(trace_dir + '/*'))

    total_proofs = 0
    verified_proofs = 0
    
    for file in tqdm(files):
        trace = pickle.load(open(file, 'rb'))
        if not trace.proof:
            logger.info(f'No proof for {trace.theorem.full_name}')
        else:
            total_proofs += 1
            try:
                res = check_file(trace)
                if res:
                    verified_proofs += 1
            except Exception as e:
                logger.warning(f'Error verifying proof of {file}: {e}')
                continue

    logger.info(f'Valid proofs: {verified_proofs}/{total_proofs}')
