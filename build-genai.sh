common_pkgs=(
  python=3
  ## base scientific libraries
  numpy
  scipy
  pandas
  natsort
  scikit-learn
  networkx
  beautifulsoup4
  polars
  # computer vision
  opencv
  # GIS pandas
  geopandas
  # symbolic math + arbitrary precision
  sympy
  mpmath
  gmp
  ## plotting libraries
  matplotlib
  plotly
  bokeh
  seaborn
  ## interaction
  ipython
  ipywidgets=8
  ipykernel
  voila
  # progress bars
  tqdm
  ## compression
  zstandard
  pyarrow
  feather-format
  ## serialization libraries
  pyyaml
  yaml
  ## read excel files from pandas
  openpyxl
  xlrd
  ## DASK for parallel/distributed computing
  dask
  ###
  # build tools (in particular for numpy + f2py)
  meson
  # snap (doesn't in conda-forge channel)
  # protein data
  gemmi
  # common sci tools
  ffmpeg
  # mamba/conda install openssl versions that will conflict
  # with base system openssl, so install openssh + rsync
  openssh
  rsync
)

genai2501_opts=(
  -c conda-forge -c pytorch -c nvidia
  "pytorch::pytorch>=2.4"
  pytorch::torchvision
  pytorch::torchaudio
  pytorch::pytorch-cuda=12.4
  openai
  langchain
  huggingface_hub
  "transformers>=4.43.1"
  "sentence-transformers>=2.2.2"
  ca-certificates
  certifi
  openssl
  einops
  accelerate
  nodejs
  plotly
  ipykernel
  chromadb
  pypdf
  tabulate
  tokenizers
  timm=1.0.9
  gradio==5.0.0
  langchain-community
  langchain-chroma
  bitsandbytes
  scalene
  diffusers
)
mamba create -p /packages/envs/genai25.01 "${genai2501_opts[@]}" "${common_pkgs[@]}"
# pip install InstructorEmbedding
# add /packages/envs/genai25.01/lib/python3.12/site-packages/gradio/frpc_linux_amd64_v0.3
