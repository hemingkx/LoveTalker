CUDA_VISIBLE_DEVICES="1"
python main.py train --plot-every=150\
					 --batch-size=128\
                     --pickle-path='tang.npz'\
                     --lr=1e-3 \
                     --env='poetry3' \
                     --epoch=50
