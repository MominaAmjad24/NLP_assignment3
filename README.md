To run the differnt translation code: 
python3 train_translation.py --model gru --embedding onehot
python3 train_translation.py --model lstm --embedding onehot
python3 train_translation.py --model lstm --embedding pretrained
python3 train_translation.py --model gru --embedding pretrained
