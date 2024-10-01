dir=$1
mkdir -p $1
scp -r sean@10.161.8.1:~/Documents/bait/$1/traces ./$1


