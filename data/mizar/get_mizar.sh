cd "$(dirname "${BASH_SOURCE[0]}")"
git clone https://github.com/JUrban/deepmath.git && cd deepmath && mv nnhpdata ../raw_data && cd .. && rm -r deepmath -f