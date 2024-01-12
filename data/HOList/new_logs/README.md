## Generating Proof Logs with HOList

- Select which source files in the Makefile for HOList
- Start a docker container with Dockerfile_check_proofs (run `environments/holist/gen_holist_data.sh`)
- Run with -it flag to get bash terminal. Then run the relevant script in src (e.g. complex or check_proofs)
- Use docker cp to copy the prooflog and theorem databases 