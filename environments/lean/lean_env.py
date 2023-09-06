from environments.lean.leanenv import LeanEnv, Action

def load_decls(path):
    with open(path) as f:
        decls = f.read().splitlines()
    return decls


if __name__ == '__main__':

    decls = load_decls('environments/lean/leanenv/declarations/mathlib_decls.log')


    for decl in decls:
        env = LeanEnv('environments/lean/lean-gym', decl=decl)

        # initial state: (state_id, goal)
        proof_state, info = env.reset(return_info=True)
        print (proof_state)

# Provide name of theorem we would like to prove
    # env = LeanEnv('environments/lean/lean-gym', decl='int.prime.dvd_mul')
    #
    # # initial state: (state_id, goal)
    # proof_state, info = env.reset(return_info=True)
    #
    # # action is a tuple: (state_id, tactic)
    # action = Action(proof_state.id, 'intros')
    #
    # # proof step
    # proof_state, reward, done, info = env.step(action)
    # print(proof_state.state)
    #
    # # Complete proof
    # proof_state, reward, done, info = env.step(
    #     (Action(proof_state.id, 'apply (nat.prime.dvd_mul hp).mp')))
    #
    # print(proof_state.state)
    #
    # proof_state, reward, done, info = env.step(
    #     Action(proof_state.id, 'rw ← int.nat_abs_mul'))
    #
    # print(proof_state.state)
    #
    # # proof_state, reward, done, info = env.step(
    # #     Action(proof_state.id, 'exact int.coe_nat_dvd_left.mp h'))
    #
    #
    # # proof_state, reward, done, info = env.step(
    # #     Action(proof_state.id, 'apply int.coe_nat_dvd_left.mp'))
    #
    #
    # # print(proof_state.state)
    #
    # proof_state, reward, done, info = env.step(
    #     Action(proof_state.id, 'exact int.coe_nat_dvd_left.mp h'))
    #
    # print(proof_state.state)
    #
    # # If proof is complete, the returned state is "no goals"