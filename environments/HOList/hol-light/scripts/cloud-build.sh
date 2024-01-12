#!/bin/bash
set -exuo pipefail
ulimit -s 65536

apt-get update
apt-get install -y build-essential

mkdir -p build
pushd build


# TODO: vendor dependency sources in GCS for stability
curl -L http://caml.inria.fr/pub/distrib/ocaml-4.04/ocaml-4.04.2.tar.gz -o ocaml-4.04.2.tar.gz
tar -xzf ocaml-4.04.2.tar.gz
pushd ocaml-4.04.2
./configure
make world.opt
make install
popd

curl -L https://github.com/camlp5/camlp5/archive/rel701.tar.gz -o camlp5.tar.gz
tar -xzf camlp5.tar.gz
pushd camlp5-rel701
./configure
make world.opt
make install
cp {main/pcaml,main/quotation,etc/pa_reloc,meta/q_MLast}.{cmi,cmx,o} $(camlp5 -where)
popd

# TODO: find a convenient way for users to access hol-light after it
# is built.
curl -L https://github.com/brain-research/hol-light/archive/master.tar.gz -o hol-light.tar.gz
tar -xzf hol-light.tar.gz
pushd hol-light-master
make depend
make native
popd

popd
