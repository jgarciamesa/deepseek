wget https://github.com/conda-forge/miniforge/releases/download/25.1.1-0/Miniforge3-25.1.1-0-Linux-x86_64.sh
bash Miniforge3-25.1.1-0-Linux-x86_64.sh 
conda config --set auto_activate_base false
source ~/.bashrc
bash build-genai.sh 
