# Requires conda, poetry
# Installs torch with both CPU and GPU support, so help me god

conda env create -f environment.yaml
conda activate nlp_attacks
poetry install

# Note that if rerunning poetry install, this command needs to be rerun
# afterwards too, otherwise there will be no GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

mypy --install-types
