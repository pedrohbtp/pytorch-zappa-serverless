python3.6 -m venv python3.6_venv
source python3.6_venv/bin/activate
pip3 install -r requirements.txt
cp zappa_settings.rename.json zappa_settings.json
# Download example models
mkdir models
mkdir aws_lstm
wget http://files.fast.ai/models/wt103_v1/lstm_wt103.pth -P ./models/awd_lstm
wget http://files.fast.ai/models/wt103_v1/itos_wt103.pkl -P ./models/awd_lstm
