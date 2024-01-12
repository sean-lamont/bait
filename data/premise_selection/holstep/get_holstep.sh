cd "$(dirname "${BASH_SOURCE[0]}")"
wget http://cl-informatik.uibk.ac.at/cek/holstep/holstep.tgz
tar zxvf holstep.tgz
mv holstep/* raw_data
rm -r holstep